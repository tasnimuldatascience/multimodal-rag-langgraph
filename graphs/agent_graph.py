from langgraph.graph import StateGraph, END
from agents.retrieval_agent import retrieval_agent
from agents.generation_agent import generation_agent

def create_agent_graph():
    graph = StateGraph(dict)

    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("generation", generation_agent)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", END)

    return graph.compile()
