# Adaptive RAG Pipeline 
A retrieval system that dynamically routes queries to optimize answer quality and reduce hallucinations.

Most simple RAG systems fail due to a lack of method diversity. This Adaptive RAG Pipeline address this issue. It attempts multiple query translation methods to find the best way of transforming a query to improve retrieval.

This RAG Systems has been intalized with documents about Greek Mythology and The US Constitution. 
Furthermore, this RAG System allows the user to upload any website url or pdf, which is then stored into vectorbase and can be retrieved from. However, these uploaded documents are not permanent and are only accessible during each user's session. 

![RAG System Graph](RAG_graph.png)


This Adaptive RAG System is quite greedy. Using an LLM's own logic, it selects the best query transformation for a given user query. However, if that method fails to provide sufficient context to answer the query, it will try another method.

## How It Works
1. Pick the best query translation method for the given query.
2. Retrieve documents, rank them by relevance, and provide them to an LLM.
3. If the LLM determines the documents aren't sufficient to answer the query, it returns to step 1 and selects another translation method.
4. The pipeline ends only once the LLM decides it has enough context to generate a good answer, or all 3 query translation methods have been exhausted.



## Evaluation Scores
Evaluations were conducted using DeepEval across 49 domain-specific questions
| Success Rate | Average Faithfulness Score | Average Answer Relevancy Score |
| -------- | -------- | -------- |
| 98.96%   | 0.9898   | 0.9864   |


## Packages Used

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [DeepEval](https://docs.confident-ai.com/)
- [HuggingFace](https://huggingface.co/)
- `langchain-openai`

## How to Run Locally
```bash
git clone "https://github.com/VakeesanM/RAG-System-with-Evaluation.git"
pip install -r requirement.txt
streamlit run "RAGSystem\app.py"
```
