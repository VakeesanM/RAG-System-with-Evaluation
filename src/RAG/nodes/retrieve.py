from RAG.state import RAGState



def retrieve(state: RAGState) -> RAGState:
    retriever = state['retriever'] 
    full_context = []
    for query in state['translated_query']:
        context = retriever.invoke(query)
        full_context.extend(context) 

    if len(state['translated_query']) > 1:
        full_context = find_unique_chunks(full_context)

    
    state['context'] = full_context
    return state


def find_unique_chunks(all_chunks):
    unqiue_chunks = {}
    for doc in all_chunks:
        unqiue_chunks[doc.page_content] = doc
    return list(unqiue_chunks.values())
