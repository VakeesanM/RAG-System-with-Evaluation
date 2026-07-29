from setup.setup import initial_chunk
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from RAG.initilize_system import initilize_rag_graph


class RAGSystem():
    def __init__(self):
        embedder = OpenAIEmbeddings(model='text-embedding-3-small')
        self.vectorbase = FAISS.load_local("RAG/faiss_index", embedder, allow_dangerous_deserialization=True)
        self.chunks = initial_chunk()
        
        self.update_retiever()

        self.encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.system = initilize_rag_graph()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def update_database(self, documents):
        chunks = self.splitter.split_documents(documents)
        self.vectorbase.add_documents(chunks)
        self.chunks = self.chunks + chunks
        self.update_retiever()

    def update_retiever(self):
        dense_retriever = self.vectorbase.as_retriever(search_kwargs={"k": 5})
        spare_retriever = BM25Retriever.from_documents(self.chunks)

        self.retriever = EnsembleRetriever(
            retrievers=[dense_retriever, spare_retriever],
            weights=[0.6, 0.4]
        )

    def chunk_pdf(self, pdf):
        documents = PyPDFLoader(pdf).load()
        self.update_database(documents)

    def chunk_website(self, website_url):
        documents = WebBaseLoader(website_url).load()
        self.update_database(documents)

    def invoke(self, query):
        result = self.system.invoke({
            "query": query,
            "retriever": self.retriever,
            'encoder': self.encoder,
            'translation_attempts': []
        })
        return result['final_output']

    def stream(self, query):
        result = self.invoke(query)
        for token in result:
            yield token