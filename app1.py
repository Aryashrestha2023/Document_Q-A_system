import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from textblob import TextBlob
import joblib



# Load API key and configure Google Generative AI
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_uploaded_text(files):
    text = ""
    for file in files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.type == "text/plain":
            text += file.read().decode("utf-8")  # Decode bytes to string
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handling chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    answer = response["output_text"]

    
    return answer

def summarize_text(text):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    prompt = f"Summarize the following pdf :\n\n{text}\n\nSummary:"
    response = model.invoke(prompt)
    summary = response.content


    return summary

def main():
    st.header("Q&A RAG System")

    # Display chat history
    with st.container():
        st.subheader("📜 Chat History")
        with st.expander("View Conversation", expanded=True):
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")
                st.markdown("---")

    user_question = st.text_input("Ask a Question from the PDF Files")


    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu:")
        # pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

        pdf_docs = st.file_uploader(
            "Upload your PDF or TXT Files",
            accept_multiple_files=True,
            type=["pdf", "txt"]
        )


        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # raw_text = get_pdf_text(pdf_docs)
                raw_text = get_uploaded_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state["raw_text"] = raw_text  # Save for summarization
                st.success("PDF Processed ✅")

    # Summarize PDF in the main screen
    if "raw_text" in st.session_state and st.button("Summarize PDF", key="summarize_button"):
        with st.spinner("Summarizing..."):
            summary, emotion, sentiment = summarize_text(st.session_state["raw_text"])
            st.session_state["summary"] = summary

            # Display the PDF summary with emotion and sentiment
            st.subheader("📄 PDF Summary:")
            st.write(st.session_state["summary"])


if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []

if __name__ == "__main__":
    main()
