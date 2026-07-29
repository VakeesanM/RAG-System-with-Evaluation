from RAG.state import RAGState

def initilize(state: RAGState):
    state['translated_query'] = ''
    state['context'] = []
    state['final_output'] = ''


    return state

