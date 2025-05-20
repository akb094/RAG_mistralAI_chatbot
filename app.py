# app.py
import streamlit as st
from build_rag import qa_chain

st.set_page_config(page_title="COVID-19 RAG Chatbot", page_icon="🧠")
st.title("🧠 COVID-19 RAG Chatbot")

query = st.text_input("Ask a question:")

if query:
    result = qa_chain({"query": query})
    st.write("### Answer")
    st.write(result['result'])

    with st.expander("Sources"):
        for doc in result["source_documents"]:
            st.markdown(f"- {doc.page_content[:200]}...")
