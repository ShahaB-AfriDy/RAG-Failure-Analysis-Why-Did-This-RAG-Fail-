from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

# -----------------------------
# 1. Load environment and API
# -----------------------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# -----------------------------
# 2. Set up embedding model
# -----------------------------
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key,
    embedding_kwargs={"output_dimensionality": 3072}
)

# -----------------------------
# 3. Load corpus
# -----------------------------
loader = TextLoader("corpus.txt", encoding="utf-8")
documents = loader.load()

# -----------------------------
# 4. Text splitting (slightly larger chunk size)
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,  
    chunk_overlap=5
)
docs = text_splitter.split_documents(documents)

# -----------------------------
# 5. Vectorstore creation
# -----------------------------
df_path = r"D:\RAG Failure Analysis\chroma_db"
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=os.path.join(df_path, "chroma_db1")
)


# -----------------------------
# 6. Retriever 
# -----------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# 7. LLM setup
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=google_api_key
)

# -----------------------------
# 8. Query and retrieve context
# -----------------------------
query = "What time does the library close?"
retrieved_docs = retriever.invoke(query)
context = "\n".join([doc.page_content for doc in retrieved_docs])

# -----------------------------
# 9. Prompt
# -----------------------------
prompt = f"""
Answer the question using only the context below.
If the answer is not clearly stated in the context, say "Insufficient information."

Context:
{context}

Question:
{query}
"""

# -----------------------------
# 10. Generate answer
# -----------------------------
response = llm.invoke(prompt)
print(f"Question: {query}")
print("\nFinal Answer:")
print(response.content)

# Optional: inspect retrieved chunks
print("\nRetrieved Context Chunks:")
print(context.split("\n"))
