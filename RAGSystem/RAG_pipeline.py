from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel 
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker



class ChoiceOutput(BaseModel):
    method: int
class MultiQueryOutput(BaseModel):
    questions: List[str]
class GenOutput(BaseModel):
    output: Union[str, int]
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)

### Query Transformation Prompts, LLMs, pipeline

rewritePrompt = ChatPromptTemplate([
    ("system", """You are a query rewriting agent in a RAG pipeline.
    If you think the inputted query is unclear or hard to understand then your task is to rewrite the given query so that it is:
    - Free of grammar and spelling mistakes
    - Clear and unambiguous
    - Well-structured and easy to understand

    Return only the rewritten query, with no explanation or preamble."""),
    ("human", "Query: {query}")
])

multiQueryPrompt = ChatPromptTemplate([
    ("system", """You are a query decomposition agent in a RAG pipeline.

    Your task is to break down the given query into multiple focused sub-queries. Each sub-query should:
    - Address a distinct aspect of the original query
    - Act as a stepping stone toward answering the main query
    - Be self-contained and independently retrievable

    Return a list of sub-queries only, with no explanation or preamble."""),
    ("human", "Query: {query}")
])

stepBackPrompt = ChatPromptTemplate([
    ("system", """You are a step-back prompting agent in a RAG pipeline.

    Your task is to transform the given query into a broader, more abstract version of it. The step-back query should:
    - Generalize the original query to a higher conceptual level
    - Capture the underlying topic or principle being asked about
    - Help retrieve background context that supports answering the original query

    Return only the step-back query, with no explanation or preamble."""),
    ("human", "Query: {query}")
])


rewriteChain = rewritePrompt| llm
multiQueryChain = multiQueryPrompt | llm.with_structured_output(MultiQueryOutput)
stepBackChain = stepBackPrompt|llm

# LLM that uses Ai's logic to chose a choice.
queryTranslationChoicePrompt = ChatPromptTemplate([
    ("system", """You are a Query Translation Router inside an Adaptive RAG system. Your sole responsibility is to select the most appropriate Query Translation method for a given query.

    ## Available Query Translation Methods

    1. **RAG Rewrite**
    - Reformulates the query for better retrieval alignment
    - Best for: queries with typos, ambiguous phrasing, poor grammar, or unclear intent
    - Example triggers: short vague queries, colloquial language, pronouns without clear referents

    2. **Multi-Query**
    - Generates multiple query variants to broaden retrieval coverage
    - Best for: complex or multi-faceted queries touching several distinct concepts
    - Example triggers: "compare X and Y", "what are the causes and effects of Z", compound questions

    3. **StepBack**
    - Abstracts the query to a higher-level concept to improve context retrieval
    - Best for: highly specific or narrow queries where the answer lives in broader background material
    - Example triggers: queries about a specific instance of a general concept, deep technical questions

    4. **No Method — LAST RESORT ONLY**
    - Use this ONLY if ALL three methods (1, 2, and 3) have already been attempted.
    - NEVER select 4 if any method remains untried — an untried method always has a chance of succeeding.
    - Do NOT select 4 because a query seems out-of-scope or unlikely to match. You cannot know what is in the document store.

    ## Decision Rules
    - ALWAYS prefer an untried method over giving up. Assume the document store may contain relevant information until all methods are exhausted.
    - You will receive a list of previously attempted methods. Never repeat a method that has already been tried.
    - Work through methods in order of fit: pick the best match for the query first, then fall back to the next best on retries.
    - Method 4 is only valid when the previously attempted list contains all of [1, 2, 3].

    ## Retry Guidance
    - First attempt: pick whichever of 1, 2, or 3 best fits the query.
    - Second attempt: pick the next most applicable untried method — do not give up.
    - Third attempt: use whichever method remains untried.
    - Only after all three have been tried: select 4.

    ## Output Format
    Return ONLY the single integer associated with your chosen method (1, 2, 3, or 4). No explanation, no extra text.
    """),
    ("human", "Query: {query}\nPreviously Attempted Methods: {transformation_attempts}")
])

translationChoiceChain = queryTranslationChoicePrompt | llm.with_structured_output(ChoiceOutput)

class RAGState(TypedDict):
    query: str
    vectorstore: FAISS
    encoder: HuggingFaceCrossEncoder # I put it here so that the encoder can be intizlied inside of app.py. Or else it starts repeteadly reinitializing over and over agin. 

    #Everything Else here can be ignored. 
    transformation_attempts: List[int]
    rewritten_query: str
    subqueries: List[str]
    stepBack: str
    context: List[Document] 
    finalOutput: Union[str, int]
    

def choice(state: RAGState) -> RAGState:
    state['rewritten_query'] = ''
    state["subqueries"] = []
    state['stepBack'] = ''
    state['context'] = []
    state['finalOutput'] = ''
    result = translationChoiceChain.invoke({
        "query": state['query'],
        "transformation_attempts": state['transformation_attempts']
    })
    state['transformation_attempts'].append(result.method)
    if result.method == 4:
        state['finalOutput'] = "This query can't be answered based on given documents"
    return state

def enforceChoice(state: RAGState):
    return state['transformation_attempts'][-1]

def ragRewrite(state: RAGState) -> RAGState:
    result = rewriteChain.invoke({
        "query": state['query']
    })
    state['rewritten_query'] = result.content
    return state

def multiQuery(state: RAGState) -> RAGState:
    result = multiQueryChain.invoke({
        "query": state['query']
    })
    state['subqueries'] = result.questions
    return state

def stepback(state: RAGState) ->  RAGState:
    result = stepBackChain.invoke({
        "query": state['query']
    })
    state['stepBack']= result.content
    return state

def retrieve(state: RAGState) -> RAGState:
    retriever = state['vectorstore'].as_retriever(search_kwargs={"k": 8})
    if state['transformation_attempts'][-1] == 1:
        state['context'] = (retriever.invoke(state['rewritten_query']))
    elif state['transformation_attempts'][-1] == 2:
        for subqueries in state['subqueries']:
            state['context'] = state['context'] + (retriever.invoke(subqueries))

        # deleting dups
        unqiue_chunks = {}
        for doc in state['context']:
            unqiue_chunks[doc.page_content] = doc
        state['context'] = list(unqiue_chunks.values())
    elif state['transformation_attempts'][-1] == 3:
        state['context'] = (retriever.invoke(state['stepBack']))

    return state
        

def rerank(state: RAGState) -> RAGState:
    reRanker= CrossEncoderReranker(model=state['encoder'], top_n=5)
    state['context'] = reRanker.compress_documents(state['context'], state['query'])
    return state

def finish(state: RAGState):
    if isinstance(state['finalOutput'], str) or (len(state["transformation_attempts"]) > 2):
        return 1
    elif isinstance(state['finalOutput'], int):
        return 0
    
    

# Generation LLM

genPrompt = ChatPromptTemplate([
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

genChain = genPrompt | llm.with_structured_output(GenOutput)

def gen(state: RAGState) -> RAGState:
    context = ("\n\n".join(doc.page_content for doc in state['context']))
    result = genChain.invoke({
        "query": state["query"],
        'context': context
    })

    state['finalOutput'] = result.output

    return state

graph = StateGraph(RAGState)
#Layer 1 - Base
graph.add_node("Chose Translation", choice)

#Layer 2 - Translation
graph.add_node("RAG Rewrite", ragRewrite)
graph.add_node("Mutli Query", multiQuery)
graph.add_node("stepback", stepback)

#Layer 3 - Reterival and Ranking
graph.add_node("retrieve", retrieve)
graph.add_node("rerank", rerank)

#Layer 4 - Generation
graph.add_node("gen", gen)


graph.add_edge(START, "Chose Translation")
graph.add_conditional_edges(
    "Chose Translation",
    enforceChoice,
    {
        1: "RAG Rewrite",
        2: "Mutli Query",
        3: "stepback",
        4: END
    }
)

graph.add_edge("RAG Rewrite", "retrieve")
graph.add_edge("Mutli Query", "retrieve")
graph.add_edge("stepback", "retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "gen")

graph.add_conditional_edges(
    "gen",
    finish,
    {
        0: "Chose Translation",
        1: END
    }
)


app = graph.compile()

def ragInvoke(query: str, vectorstore: FAISS, encoder: HuggingFaceCrossEncoder):
    result = app.invoke({
        "query": query,
        "vectorstore": vectorstore,
        "encoder": encoder,
        'transformation_attempts': []
    })
    if isinstance(result['finalOutput'], int):
        return "I can't answer this query based on the documents provided!"

    return result['finalOutput']

def testing(query: str, vectorstore: FAISS, encoder: HuggingFaceCrossEncoder):
    result = app.invoke({
        "query": query,
        "vectorstore": vectorstore,
        "encoder": encoder,
        'transformation_attempts': []
    })

    context = "\n\n".join(doc.page_content for doc in result['context'])
    return result['finalOutput'], context


#img = (app.get_graph().draw_mermaid_png())
#with open("RAG_graph.png", 'wb') as f:
    #f.write(img)