# build_rag.py
import os
import requests
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from mistral_custom import MistralAI

# Ensure the API key is set
assert os.getenv("MISTRAL_API_KEY"), "❌ Please set MISTRAL_API_KEY in your environment."

# Load COVID dataset
url = "https://raw.githubusercontent.com/deepset-ai/COVID-QA/master/data/question-answering/COVID-QA.json"
response = requests.get(url)
data = response.json()

# Extract Q&A pairs
qa_pairs = []
for article in data["data"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]["text"] if qa["answers"] else "No answer provided"
            qa_pairs.append({"question": question, "answer": answer, "context": context})

# Convert to LangChain documents
documents = [
    Document(page_content=f"Context: {item['context']}\nQ: {item['question']}\nA: {item['answer']}")
    for item in qa_pairs[:300]  # Limit for speed
]

# Text splitting
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = []
for doc in documents:
    docs.extend(splitter.split_documents([doc]))

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector store
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# Custom Mistral LLM
llm = MistralAI(model="mistral-tiny", temperature=0)

# QA chain
qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
