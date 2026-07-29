from RAG.state import RAGState
from RAG.base_model.multi_query_output import MultiQueryOutput

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

multi_query_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5).with_structured_output(MultiQueryOutput)
multi_query_prompt = ChatPromptTemplate([
    ("system", """You are a query decomposition agent in a RAG pipeline.

    Your task is to break down the given query into multiple focused sub-queries. Each sub-query should:
    - Address a distinct aspect of the original query
    - Act as a stepping stone toward answering the main query
    - Be self-contained and independently retrievable

    Return a list of sub-queries only, with no explanation or preamble."""),
    ("human", "Query: {query}")
])

multi_query_chain = multi_query_prompt | multi_query_llm


def multi_query(state: RAGState) -> RAGState:
    translated_queries = multi_query_chain.invoke({
        'query': state['query']
    }).questions

    state['translated_query'] = translated_queries
    return state


