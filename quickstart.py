# Import necessary libraries
import math  # Standard library for mathematical functions
import types  # For type checking
import uuid  # For generating unique identifiers

# Import LangChain components for language models and embeddings
from langchain.chat_models import init_chat_model  # Initialize chat language models
from langchain.embeddings import init_embeddings  # Initialize embedding models
from langgraph.store.memory import InMemoryStore  # Import in-memory storage for LangGraph

# Import our custom agent creation function and utility functions
from langgraph_bigtool import create_agent
from langgraph_bigtool.utils import (
    convert_positional_only_function_to_tool  # Utility to convert Python functions to LangChain tools
)

# Collect mathematical functions from the `math` built-in library and convert them to tools
all_tools = []
for function_name in dir(math):  # Loop through all attributes in the math module
    function = getattr(math, function_name)  # Get the attribute
    if not isinstance(
        function, types.BuiltinFunctionType
    ):
        continue  # Skip if not a built-in function
    # Convert the function to a tool format that can be used with LangChain
    # The math library uses positional-only parameters, so we need a special conversion
    if tool := convert_positional_only_function_to_tool(
        function
    ):
        all_tools.append(tool)

# Create a registry of tools with unique IDs
# This maps randomly generated UUIDs to tool instances
# The tool_registry will be used by the agent to look up and execute tools
tool_registry = {
    str(uuid.uuid4()): tool  # Generate a random UUID as the key for each tool
    for tool in all_tools
}

# Initialize embedding model for semantic search
# This will be used to find relevant tools based on natural language descriptions
embeddings = init_embeddings("openai:text-embedding-3-small")

# Set up an in-memory store with vector indexing capabilities
# This store will hold tool metadata and enable semantic search over tools
store = InMemoryStore(
    index={
        "embed": embeddings,  # The embedding model to use
        "dims": 1536,  # Dimension of the embeddings
        "fields": ["description"],  # Which fields to embed for search
    }
)

# Store each tool's metadata in the vector store
# The key is the tool_id, and we store a description that combines the tool name and description
for tool_id, tool in tool_registry.items():
    store.put(
        ("tools",),  # The namespace/collection for storing tools
        tool_id,  # The unique identifier for this tool
        {
            "description": f"{tool.name}: {tool.description}",  # Format the description for the tool
        },
    )

# Initialize the language model that will power the agent
llm = init_chat_model("openai:gpt-4o-mini")

# Create the agent using our builder pattern
# This sets up a LangGraph agent with the specified LLM and tools
builder = create_agent(llm, tool_registry)
agent = builder.compile(store=store)  # Compile the agent with the configured store

# The variable 'agent' is left at the end of the file
# In Python, the last expression in a file or cell can be implicitly returned
# This is particularly useful in notebook environments like Jupyter
# When this file is imported or executed in a notebook, the 'agent' variable
# will be available for use, allowing immediate interaction with the agent
# It's a common pattern in interactive Python environments to have the
# object you want to work with as the last line
agent