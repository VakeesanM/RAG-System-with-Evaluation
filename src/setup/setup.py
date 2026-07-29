from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings



def initial_chunk():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    Constitution = WebBaseLoader("https://www.senate.gov/about/origins-foundations/senate-and-constitution/constitution.htm").load()
    Constitution_docs = splitter.split_documents(Constitution)
    
    Greek_Mythos = PyPDFLoader(r"src/RAG/setup/Greek_Mythology.pdf").load()
    Greek_Mythos_docs = splitter.split_documents(Greek_Mythos)
    
    for doc in Constitution_docs:
        doc.metadata["source"] = "Constitution"
    for doc in Greek_Mythos_docs:
        doc.metadata["source"] = "Greek Mythology"

    Combined_doc = Constitution_docs + Greek_Mythos_docs

    return Combined_doc

def update_chunks(old_chuncks, new_chunkcs):
    return old_chuncks + new_chunkcs

def intalize_database(chunks):
    Embedder = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorDB = FAISS.from_documents(chunks, embedding=Embedder)
    vectorDB.save_local("faiss_index")


import os
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    os.getenv("OPENAI_API_KEY")

    chunks = initial_chunk()
    intalize_database(chunks)




