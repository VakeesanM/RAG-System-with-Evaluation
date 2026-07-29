from RAG.state import RAGState

def enforce_choice(state: RAGState):
    return state['translation_attempts'][-1]