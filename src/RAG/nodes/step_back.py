from RAG.state import RAGState


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


step_back_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)
step_back_prompt = ChatPromptTemplate([
    ("system", """You are a step-back prompting agent in a RAG pipeline.

    Your task is to transform the given query into a broader, more abstract version of it. The step-back query should:
    - Generalize the original query to a higher conceptual level
    - Capture the underlying topic or principle being asked about
    - Help retrieve background context that supports answering the original query

    Return only the step-back query, with no explanation or preamble."""),
    ("human", "Query: {query}")
])

step_back_chain = step_back_prompt | step_back_llm


def step_back(state: RAGState) -> RAGState:
    translated_query = step_back_chain.invoke({
        'query': state['query']
    }).content
    state['translated_query'] = translated_query
    return state
