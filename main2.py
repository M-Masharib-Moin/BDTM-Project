import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import string

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from htmlTemplates import css, user_template, bot_template
load_dotenv()

# Utility: Clean up text
def filter_text(text):
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, text))

# Step 1: Extract and chunk PDF text
def get_chunks(pdf_docs):
    splits = []
    for pdf_doc in pdf_docs:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_doc.read())
                temp_file.close()

                loader = PyPDFLoader(temp_file.name)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=150
                )
                document_chunks = text_splitter.split_documents(documents)

                # Filter short or empty chunks
                filtered_chunks = [
                    chunk for chunk in document_chunks
                    if len(chunk.page_content.strip()) > 50
                ]
                splits.extend(filtered_chunks)

                os.unlink(temp_file.name)
        except Exception as e:
            st.error(f"Error processing {pdf_doc.name}: {e}")
    return splits

# Step 2: Embed and store
persist_directory = './chroma'

def get_vectorstore(splits):
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb

# Step 3: Conversation chain setup
def get_conversation_chain(vectordb):
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. Keep answers concise (max 3 sentences).
End every response with: "Thanks for asking!".
{context}
Question: {question}
Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    retriever = vectordb.as_retriever()

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa

# Step 4: Handle Q&A
def handle_userinput(user_question, qa):
    if qa is not None:
        result = qa.invoke({"question": user_question})
        answer = result["answer"]
    else:
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        prompt = f"""Answer the following question concisely in 3 sentences or less.
Question: {user_question}
Answer:"""
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", str(answer)), unsafe_allow_html=True)

# Step 5: Streamlit UI
def main():
    load_dotenv()
    st.set_page_config(page_title="Study Snap", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    # Session state setup
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = None
    if "status" not in st.session_state:
        st.session_state.status = "Not started"

    st.header("Study Snap 📝")
    user_question = st.text_input("Enter a question about your documents or general topic:")

    if user_question:
        handle_userinput(user_question, st.session_state.conversation)

    with st.sidebar:
        st.subheader("Upload your documents")
        st.session_state.pdf_docs = st.file_uploader(
            "Upload PDFs here and Click Enter",
            accept_multiple_files=True
        )

        if st.button("Enter"):
            if st.session_state.pdf_docs:
                with st.spinner("Processing your documents..."):
                    st.session_state.status = "Reading documents..."
                    text_chunks = get_chunks(st.session_state.pdf_docs)

                    if not text_chunks:
                        st.error("No readable content found in PDFs.")
                        return

                    st.session_state.status = "Creating vectorstore..."
                    vectorstore = get_vectorstore(text_chunks)

                    st.session_state.status = "Setting up conversation chain..."
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Vectorstore created and chat is ready.")
            else:
                st.warning("Please upload at least one PDF.")

        st.caption(f"Status: {st.session_state.get('status', 'Idle')}")

if __name__ == "__main__":
    main()
