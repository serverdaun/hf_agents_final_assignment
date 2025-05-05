import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph, MessagesState
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools import wiki_search, tavily_search, arxiv_search, add, subtract, multiply, divide, power, sqrt, modulus


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

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
    modulus
]

def build_agent():
    # Define llm from Hugging Face
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        huggingfacehub_api_token=HF_TOKEN
    )

    # Define chat interface and the tools
    chat = ChatHuggingFace(llm=llm, verbose=True)
    chat_w_tools = chat.bind_tools(TOOLS)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [chat_w_tools.invoke(state["messages"])]}


    builder = StateGraph(MessagesState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()