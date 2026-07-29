from RAG.state import RAGState
from RAG.base_model.gen_output import GenOutput 

from langchain_openai import ChatOpenAI
from langchain_classic.prompts import ChatPromptTemplate


generation_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5).with_structured_output(GenOutput)
generation_prompt = ChatPromptTemplate([
    ("system", """You are the answer generation agent in the final step of a RAG pipeline.

    You will be given a query and a set of retrieved context passages. Your task is to answer the query using only the provided context. Follow these rules:
    - Answer using only information present in the context — do not use prior knowledge
    - If the context does not contain enough information to answer the query, return the integer 1.
    - Keep answers concise and to the point.
    - Do not reference the context directly (e.g. avoid phrases like "based on the context...")
    - Some context chunks may contain numbers in Roman Numerals.  
    - Your Answer should always be in full sentences.
    only return a String, if you can answer the question. If you can't answer it, you must return an Integer. 
    """),
    ("human", "Query: {query}\n\nContext:\n{context}")
])

generation_chain = generation_prompt | generation_llm

def generate(state: RAGState) -> RAGState:
    result = generation_chain.invoke({
        'query': state['query'],
        'context': state['context']
    }).output

    state['final_output'] = result
    return state

