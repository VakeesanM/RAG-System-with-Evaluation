from RAG.state import RAGState
from RAG.nodes.initilization import initilize
from RAG.base_model.choice_output import ChoiceOutput 

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


choice_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5).with_structured_output(ChoiceOutput)
choice_reasoning_prompt = ChatPromptTemplate([
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
    ("human", "Query: {query}\nPreviously Attempted Methods: {translation_attempts}")
])

translation_choice_chain = choice_reasoning_prompt | choice_llm

def chose_translation(state: RAGState) -> RAGState:
    state = initilize(state)

    choice = translation_choice_chain.invoke({
        "query": state['query'],
        'translation_attempts': state['translation_attempts']
    }).method

    state['translation_attempts'].append(choice)
    if choice == 4:
            state['finalOutput'] = "This query can't be answered based on given documents"


    return state
