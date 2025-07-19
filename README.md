# Document_Q-A_system

This project is a **Streamlit-based Document Q&A system** that processes **PDF** and **TXT** files. It allows users to **ask questions** about uploaded documents and receive **AI-generated answers** using **Google Gemini** models. It also supports **summarization of PDF content** and maintains a **chat history** for user queries.

---

## **Features**
- **Upload PDFs or TXT files** and process their content.
- **Vector-based semantic search** using FAISS and Google Generative AI Embeddings.
- **Ask questions** about uploaded documents with detailed AI-generated responses.
- **Summarize uploaded PDF content** using Google Gemini.
- **Chat history** to review past questions and answers.
- **Sentiment analysis (via TextBlob)** for deeper document understanding.

---

## **Tech Stack**
- **Frontend:** [Streamlit](https://streamlit.io/)
- **LLM & Embeddings:** [LangChain](https://www.langchain.com/), [Google Generative AI (Gemini)](https://ai.google/), FAISS
- **PDF Parsing:** PyPDF2
- **Environment Management:** `python-dotenv`

