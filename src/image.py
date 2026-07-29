import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")

from RAG.initilize_system import initilize_rag_graph


app = initilize_rag_graph()
img = (app.get_graph().draw_mermaid_png())
with open("RAG_graph.png", 'wb') as f:
    f.write(img)
