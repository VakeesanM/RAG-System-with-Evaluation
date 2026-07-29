from RAG.state import RAGState

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


rag_rerwrite_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)
rag_rewrite_prompt = ChatPromptTemplate([
    ("system", """You are a query rewriting agent in a RAG pipeline.
    If you think the inputted query is unclear or hard to understand then your task is to rewrite the given query so that it is:
    - Free of grammar and spelling mistakes
    - Clear and unambiguous
    - Well-structured and easy to understand

    Return only the rewritten query, with no explanation or preamble."""),
    ("human", "Query: {query}")
])

rag_rewrite_chain = rag_rewrite_prompt | rag_rerwrite_llm


def rag_rewrite(state: RAGState) -> RAGState:
    translated_query = rag_rewrite_chain.invoke({
        'query': state['query']
    }).content

    state['translated_query'] = translated_query

    return state



