import streamlit as st
from RAG_pipeline import ragInvoke
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import os, tempfile
from dotenv import load_dotenv
import pandas as pd
load_dotenv()

os.getenv("OPENAI_API_KEY")

embedder = OpenAIEmbeddings(model='text-embedding-3-small')
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = FAISS.load_local( "faiss_index", embedder, allow_dangerous_deserialization=True)
    st.session_state.encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    st.session_state.df = pd.read_csv(r"RAGSystem\evaluation_dataset\Cleaned_Eval_data.csv")

st.set_page_config(page_title="Adaptive RAG App Demo", page_icon="🖥️", layout="wide")
st.title("Adaptive RAG APP - Demo \n " 
"By Vakeesan")

st.header("Upload Documents to RAG System")
url = st.text_input("Enter Link to Document:  ", )
if st.button("Submit Website URL"):
    with st.spinner("Loading Document", show_time=False):
        documents = WebBaseLoader(url).load()
    with st.spinner("Splitting Documents", show_time=False):
        chunks = splitter.split_documents(documents)
    with st.spinner("Uploading and Embedding Document int Vector Base", show_time=False):
        st.session_state.vectorstore.add_documents(chunks)
    st.write("Website has been uploaded into VectorBase")

        
pdf = st.file_uploader(label="Upload PDF into VectorStore", type=['.pdf'], max_upload_size=3)
if st.button("Submit PDF"):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp:
        temp.write(pdf.read())
        temp_path= temp.name
    with st.spinner("Loading Document", show_time=True):
        documents = PyPDFLoader(temp_path).load()
    with st.spinner("Splitting Documents", show_time=False):
        chunks = splitter.split_documents(documents)
    with st.spinner("Uploading and Embedding Document int Vector Base", show_time=False):
        st.session_state.vectorstore.add_documents(chunks)

    os.unlink(temp_path)
    st.write("PDF has been uploaded to Vectorstore!")
    


st.header("Quering The RAG System")
st.write("""
        This RAG System has been initialized with documents about Greek Mythos and The Constitution. 
        The Pipeline has been restricted via Prompts, such that it only answer questions using the documents provided, and can't use its own innate knowledge.
        So Unless you upload your documents(Via the options above), the pipeline will be unable to answer any questions about anything else that isn't Greek Mythos and The Constitution. 
         
         """)
query = st.text_area("Enter Query:")
if st.button("Sumbit Query"):
    with st.spinner("Running Rag Pipeline - This may take sometime(Around 10-20 Seconds).", show_time=True):
        st.write(ragInvoke(query, st.session_state.vectorstore, st.session_state.encoder))

st.header("RAG System Evaluation and Metrics")
st.write("Evaluation was done on 49 User Inputs using the DeepEval package for Faithfulness and Answer Relevancy. \n This RAG System, due to being an adaptive and greedy system, performs far better than basic RAG Systems")

scores = pd.DataFrame([{
    "Success Rate": st.session_state.df["success"].mean(),
    "Average Faithfulness Score": st.session_state.df["faithfulness_score"].mean(),
    "Average Answer Relevancy Score": st.session_state.df["answer_relevancy_score"].mean()
}])

st.dataframe(scores, hide_index=True)

cols = st.session_state.df.columns.to_list()
default_cols = ["input", "success", "faithfulness_score", "faithfulness_success", "answer_relevancy_score", "answer_relevancy_success"]


selected_columns = st.multiselect(
    "Select columns to display",
    options=cols,
    default=default_cols
)


st.dataframe(st.session_state.df[selected_columns], hide_index=True)