from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from functools import partial
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .graph_state import State, AgentState
from .nodes import *
from .edges import *

def create_agent_graph(llm, tools_list):
    
    # 1. Define the strict system message to prevent web hallucinations
    system_message = """You are an expert Indian Legal Assistant. 
    Answer the user's question using ONLY the provided retrieved context. 

    CRITICAL CITATION RULES:
    1. You MUST cite the source for every claim you make.
    2. ONLY use the file names provided in the context metadata (e.g., 'ipc.md', 'cpc.md').
    3. DO NOT generate, invent, or hallucinate any web URLs, hyperlinks, or external sources.
    4. If the answer is not contained in the context, say "I cannot answer this based on the provided documents."
    """

    # 2. Wrap the LLM in a ChatPromptTemplate to enforce the system instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # 3. Chain the prompt to the LLM and bind the tools
    llm_with_tools = prompt | llm.bind_tools(tools_list)
    
    tool_node = ToolNode(tools_list)

    checkpointer = InMemorySaver()

    print("Compiling agent graph...")
    agent_builder = StateGraph(AgentState)
    
    # Pass the prompt-wrapped LLM to the orchestrator
    agent_builder.add_node("orchestrator", partial(orchestrator, llm_with_tools=llm_with_tools))
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node("compress_context", partial(compress_context, llm=llm))
    agent_builder.add_node("fallback_response", partial(fallback_response, llm=llm))
    agent_builder.add_node(should_compress_context) 
    
    # ADDED: The Legal Weight Integrity node
    agent_builder.add_node("integrity_check", integrity_check)
    
    agent_builder.add_node(collect_answer)
    
    agent_builder.add_edge(START, "orchestrator")    
    
    # CHANGED: orchestrator now routes to integrity_check instead of collect_answer
    agent_builder.add_conditional_edges(
        "orchestrator", 
        route_after_orchestrator_call, 
        {"tools": "tools", "fallback_response": "fallback_response", "integrity_check": "integrity_check"}
    )
    
    agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    
    # ADDED: Conditional routing out of the integrity check
    agent_builder.add_conditional_edges(
        "integrity_check",
        route_after_integrity_check,
        {"orchestrator": "orchestrator", "collect_answer": "collect_answer"}
    )
    
    agent_builder.add_edge("collect_answer", END)
    
    agent_subgraph = agent_builder.compile()
    
    graph_builder = StateGraph(State)
    graph_builder.add_node("summarize_history", partial(summarize_history, llm=llm))
    graph_builder.add_node("rewrite_query", partial(rewrite_query, llm=llm))
    graph_builder.add_node(request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", partial(aggregate_answers, llm=llm))
    
    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)

    agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])

    print("✓ Agent graph compiled successfully.")
    return agent_graph