from typing import TypedDict, List, Union
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document


class RAGState(TypedDict):
    query: str
    retriever : EnsembleRetriever
    encoder: HuggingFaceCrossEncoder 

    translation_attempts: List[int]

    translated_query: Union[str, List[str]]
    context: List[Document] 

    final_output: Union[str, int]


