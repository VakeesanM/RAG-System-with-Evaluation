from RAG.state import RAGState
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

def rerank(state: RAGState) -> RAGState:
    ranker= CrossEncoderReranker(model=state['encoder'], top_n=5)
    
    state['context'] = ranker.compress_documents(state['context'], state['query'])

    return state