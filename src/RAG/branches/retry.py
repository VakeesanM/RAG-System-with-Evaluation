from RAG.state import RAGState

def finish(state: RAGState):
    if isinstance(state['final_output'], str) or (len(state["translation_attempts"]) > 2):
        return "END"
    elif isinstance(state['final_output'], int):
        return "Retry"