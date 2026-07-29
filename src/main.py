import streamlit as st
import os, tempfile
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
os.getenv("OPENAI_API_KEY")


from RAG.rag_system import RAGSystem
if "rag" not in st.session_state:
    st.session_state.rag = RAGSystem()
    csv_path = os.path.join(os.path.dirname(__file__), "eval", "evaluation_dataset", "Cleaned_Eval_data.csv")
    st.session_state.df = pd.read_csv(csv_path)


st.set_page_config(page_title="Adaptive RAG App Demo", page_icon="🖥️", layout="wide")
st.title("Adaptive RAG APP - Demo \n " 
"By Vakeesan")

st.write("""This RAG system is adaptive, using llm reasoning to chose which tranlastion methods to apply and wheater to reattempt reterival based on context relevance to the query. \n  \
It uses hybrid reterival methods of VectorStore Reterival and BM25 with weight of 60% and 40% respectively. \n
This adaptive approach greatly increases answer revelance, context reterival and generally results in the generation of better results at the cost of latency. 
As such, this RAG system should be used in cases where precision is important, i.e.
in cases of LLMs beings used for Medical or Law purposes, where incorrects answer could lead to patient death or incorrect sentencings.""")


st.header("Upload Documents to RAG System")

url = st.text_input("Enter Link to Document:  ", )
if st.button("Submit Website URL"):
    with st.spinner("Loading Document into Database", show_time=False):
        st.session_state.rag.chunk_website(url)
    st.write("Document has been uploaded")
        

        
pdf = st.file_uploader(label="Upload PDF into VectorStore", type=['.pdf'], max_upload_size=3)
if st.button("Submit PDF"):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp:
        temp.write(pdf.read())
        temp_path = temp.name
    with st.spinner("Loading Document into Database", show_time=True):
        st.session_state.rag.chunk_pdf(temp_path)
   

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
        st.write(st.session_state.rag.invoke(query))

st.header("RAG System Evaluation and Metrics")
st.write("Evaluation was done on 49 User Inputs using the DeepEval package for Faithfulness and Answer Relevancy. \n This RAG System, due to being an adaptive and greedy system, performs far better than basic RAG Systems at the cost of speed")


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