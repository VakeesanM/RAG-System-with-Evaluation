from RAG.state import RAGState

from RAG.nodes.reason_translation import chose_translation
from RAG.nodes.multi_query import multi_query
from RAG.nodes.rag_rewrite import rag_rewrite
from RAG.nodes.step_back import step_back
from RAG.nodes.retrieve import retrieve
from RAG.nodes.rerank import rerank
from RAG.nodes.generate import generate

from RAG.branches.query_translation import enforce_choice
from RAG.branches.retry import finish
from langgraph.graph import StateGraph, START, END


def initilize_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("Chose Translation", chose_translation)
    graph.add_node('Rag Rewrite', rag_rewrite)
    graph.add_node("Multi Query", multi_query)
    graph.add_node("Step Back", step_back)

    graph.add_node("Retrieval", retrieve)
    graph.add_node("Rerank", rerank)
    graph.add_node("Generate", generate)

    graph.add_edge(START, "Chose Translation")

    graph.add_conditional_edges(
    "Chose Translation",
    enforce_choice,
    {
        1: "Rag Rewrite",
        2: "Multi Query",
        3: "Step Back",
        4: END
    })

    graph.add_edge("Rag Rewrite", "Retrieval")
    graph.add_edge("Multi Query", "Retrieval")
    graph.add_edge("Step Back", "Retrieval")

    graph.add_edge("Retrieval", "Rerank")
    graph.add_edge("Rerank", "Generate")
    

    graph.add_conditional_edges(
        "Generate",
        finish,
        {
            "Retry": "Chose Translation",
            "END": END
        }
    )
    app = graph.compile()
    return app

if __name__ == "__main__":
    app = initilize_rag_graph()
    img = (app.get_graph().draw_mermaid_png())
    with open("RAG_graph.png", 'wb') as f:
        f.write(img)

