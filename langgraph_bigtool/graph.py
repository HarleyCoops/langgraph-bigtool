from typing import Annotated, Callable

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Send
from langgraph.utils.runnable import RunnableCallable

from langgraph_bigtool.tools import get_default_retrieval_tool


def _add_new(left: list, right: list) -> list:
    """Extend left_list with new items from right_list."""
    return left + [item for item in right if item not in set(left)]


class State(MessagesState):
    selected_tool_ids: Annotated[list[str], _add_new]


def _format_selected_tools(
    selected_tools: dict, tool_registry: dict[str, BaseTool]
) -> tuple[list[ToolMessage], list[str]]:
    tool_messages = []
    tool_ids = []
    for tool_call_id, batch in selected_tools.items():
        tool_names = [tool_registry[result].name for result in batch]
        tool_messages.append(
            ToolMessage(f"Available tools: {tool_names}", tool_call_id=tool_call_id)
        )
        tool_ids.extend(result for result in batch)

    return tool_messages, tool_ids


def create_agent(
    llm: LanguageModelLike,
    tool_registry: dict[str, BaseTool],
    *,
    limit: int = 2,
    filter: dict[str, any] | None = None,
    namespace_prefix: tuple[str, ...] = ("tools",),
    retrieve_tools_function: Callable | None = None,
    retrieve_tools_coroutine: Callable | None = None,
) -> StateGraph:
    if retrieve_tools_function is None and retrieve_tools_coroutine is None:
        retrieve_tools_function, retrieve_tools_coroutine = get_default_retrieval_tool(
            namespace_prefix, limit=limit, filter=filter
        )

    def call_model(state: State, config: RunnableConfig, *, store: BaseStore) -> State:
        selected_tools = [tool_registry[id] for id in state["selected_tool_ids"]]
        llm_with_tools = llm.bind_tools([retrieve_tools_function, *selected_tools])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    async def acall_model(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> State:
        selected_tools = [tool_registry[id] for id in state["selected_tool_ids"]]
        llm_with_tools = llm.bind_tools([retrieve_tools_function, *selected_tools])
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tool for tool in tool_registry.values())

    def select_tools(
        tool_calls: list[dict], config: RunnableConfig, *, store: BaseStore
    ) -> State:
        selected_tools = {}
        for tool_call in tool_calls:
            result = retrieve_tools_function(**tool_call["args"], store=store)
            selected_tools[tool_call["id"]] = result

        tool_messages, tool_ids = _format_selected_tools(selected_tools, tool_registry)
        return {"messages": tool_messages, "selected_tool_ids": tool_ids}

    async def aselect_tools(
        tool_calls: list[dict], config: RunnableConfig, *, store: BaseStore
    ) -> State:
        selected_tools = {}
        for tool_call in tool_calls:
            result = await retrieve_tools_coroutine(**tool_call["args"], store=store)
            selected_tools[tool_call["id"]] = result

        tool_messages, tool_ids = _format_selected_tools(selected_tools, tool_registry)
        return {"messages": tool_messages, "selected_tool_ids": tool_ids}

    def should_continue(state: State, *, store: BaseStore):
        messages = state["messages"]
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return END
        else:
            destinations = []
            for call in last_message.tool_calls:
                if call["name"] == "retrieve_tools":
                    destinations.append(Send("select_tools", [call]))
                else:
                    tool_call = tool_node.inject_tool_args(call, state, store)
                    destinations.append(Send("tools", [tool_call]))

            return destinations

    builder = StateGraph(State)

    if retrieve_tools_function is not None and retrieve_tools_coroutine is not None:
        select_tools_node = RunnableCallable(select_tools, aselect_tools)
    elif retrieve_tools_function is not None and retrieve_tools_coroutine is None:
        select_tools_node = select_tools
    elif retrieve_tools_coroutine is not None and retrieve_tools_function is None:
        select_tools_node = aselect_tools
    else:
        raise ValueError(
            "One of retrieve_tools_function or retrieve_tools_coroutine must be "
            "provided."
        )

    builder.add_node("agent", RunnableCallable(call_model, acall_model))
    builder.add_node("select_tools", select_tools_node)
    builder.add_node("tools", tool_node)

    builder.set_entry_point("agent")

    builder.add_conditional_edges(
        "agent",
        should_continue,
        path_map=["select_tools", "tools", END],
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("select_tools", "agent")

    return builder
