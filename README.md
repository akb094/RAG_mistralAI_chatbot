Retrieval-Augmented Generation (RAG) based chatbot that answers COVID-19-related questions using a combination of semantic search (FAISS) and the Mistral large language model. The app features a modern web interface built using Streamlit.
📌 What It Does

    Loads a FAQ dataset from a public source.

    Splits and embeds the text using HuggingFace's all-MiniLM-L6-v2.

    Stores the vector embeddings in a FAISS index for fast retrieval.

    Uses LangChain's RetrievalQA chain to:

        Retrieve relevant context.

        Generate answers using MistralAI (via their API).

    Provides a chat interface built with Streamlit.

🛠️ Technologies Used

    Python 3.8+

    Streamlit

    LangChain

    FAISS (Facebook AI Similarity Search)

    HuggingFace Embeddings

    Mistral AI

    QA Dataset
