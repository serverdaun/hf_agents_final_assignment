from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph, MessagesState
from langchain_openai import AzureChatOpenAI

from config import (
    MODEL_ENDPOINT,
    MODEL_KEY,
    MODEL_NAME,
    MODEL_API_VERSION,
)

from tools import (
    wiki_search,
    tavily_search,
    arxiv_search,
    add,
    subtract,
    multiply,
    divide,
    power,
    sqrt,
    modulus,
    scrape_webpage,
    analyze_image,
    is_commutative,
    commutativity_counterexample_pairs,
    commutativity_counterexample_elements,
    find_identity_element,
    find_inverses,
    transcribe_audio,
    execute_source_file,
    interact_tabular,
)

# Define tools
TOOLS = [
    wiki_search,
    tavily_search,
    arxiv_search,
    add,
    subtract,
    multiply,
    divide,
    power,
    sqrt,
    modulus,
    scrape_webpage,
    analyze_image,
    is_commutative,
    commutativity_counterexample_pairs,
    commutativity_counterexample_elements,
    find_identity_element,
    find_inverses,
    transcribe_audio,
    execute_source_file,
    interact_tabular
]


def build_agent() -> StateGraph:
    """
    Build the agent.
    Returns:
        StateGraph: The agent graph.
    """
    llm = AzureChatOpenAI(
        azure_deployment=MODEL_NAME,
        api_version=MODEL_API_VERSION,
        azure_endpoint=MODEL_ENDPOINT,
        api_key=MODEL_KEY,
    )

    chat_w_tools = llm.bind_tools(TOOLS)

    # Assistant node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [chat_w_tools.invoke(state["messages"])]}

    # Build graph
    builder = StateGraph(MessagesState)

    # Add nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))

    # Add edges
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph and return it
    return builder.compile()
