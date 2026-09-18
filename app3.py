import os
import shutil

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def get_pdf_text(pdf_docs):
    """Extract text from all uploaded PDF files."""

    text = ""

    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    return text


# ============================================================
# TEXT CHUNKING
# ============================================================

def get_text_chunks(text):
    """Split extracted PDF text into smaller chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks


# ============================================================
# CREATE FAISS VECTOR STORE
# ============================================================

def get_vector_store(text_chunks):
    """Create and save FAISS vector database."""

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# ============================================================
# GEMINI MODEL + PROMPT
# ============================================================

def get_conversational_chain():

    prompt_template = """
You are a helpful assistant answering questions about uploaded PDF documents.

Use ONLY the information provided in the CONTEXT below.

IMPORTANT:
- If the answer is present in the context, answer the question clearly.
- Do not make up information.
- If the answer is genuinely not present in the context, say exactly:
  "I don't know based on the uploaded PDF."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY
    )

    return prompt, model


# ============================================================
# ASK QUESTION
# ============================================================

def user_input(user_question):

    try:

        # ----------------------------------------------------
        # CREATE EMBEDDINGS
        # ----------------------------------------------------

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_API_KEY
        )

        # ----------------------------------------------------
        # LOAD FAISS DATABASE
        # ----------------------------------------------------

        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        # ----------------------------------------------------
        # SEARCH PDF
        # ----------------------------------------------------

        docs = new_db.similarity_search(
            user_question,
            k=6
        )

        # ----------------------------------------------------
        # CHECK RETRIEVAL
        # ----------------------------------------------------

        if not docs:
            st.warning("No relevant information was found in the PDF.")
            return

        # ----------------------------------------------------
        # CREATE CONTEXT
        # ----------------------------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # ----------------------------------------------------
        # DEBUG INFORMATION
        # ----------------------------------------------------

        with st.expander("🔍 Retrieved PDF Content", expanded=False):

            st.write(
                f"Retrieved {len(docs)} relevant sections."
            )

            st.write(
                f"Context length: {len(context)} characters"
            )

            for i, doc in enumerate(docs):

                st.markdown(
                    f"### Retrieved Section {i + 1}"
                )

                st.write(
                    doc.page_content[:2000]
                )

        # ----------------------------------------------------
        # CREATE PROMPT
        # ----------------------------------------------------

        prompt, model = get_conversational_chain()

        final_prompt = prompt.format(
            context=context,
            question=user_question
        )

        # ----------------------------------------------------
        # CALL GEMINI
        # ----------------------------------------------------

        response = model.invoke(final_prompt)

        # ----------------------------------------------------
        # HANDLE GEMINI RESPONSE
        # ----------------------------------------------------

        content = response.content

        if isinstance(content, list):

            answer_parts = []

            for block in content:

                if isinstance(block, dict):

                    if "text" in block:
                        answer_parts.append(
                            block["text"]
                        )

                else:
                    answer_parts.append(
                        str(block)
                    )

            answer = "\n".join(answer_parts)

        else:
            answer = str(content)

        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        st.markdown("### 🤖 Reply")

        st.write(answer)

    except Exception as e:

        st.error(
            f"Error while answering your question: {e}"
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # PAGE CONFIG
    # --------------------------------------------------------

    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon="📚",
        layout="wide"
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("📚 Chat with Multiple PDFs")

    st.write(
        "Upload one or more PDFs and ask questions about their content."
    )

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not GEMINI_API_KEY:

        st.error(
            "GEMINI_API_KEY was not found."
        )

        st.info(
            """
Create a `.env` file in the same folder as app3.py:

GEMINI_API_KEY=your_actual_api_key
"""
        )

        st.stop()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.header("📋 PDF Menu")

        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        process_button = st.button(
            "🚀 Submit and Process",
            use_container_width=True
        )

        # ----------------------------------------------------
        # PROCESS PDF
        # ----------------------------------------------------

        if process_button:

            if not pdf_docs:

                st.warning(
                    "Please upload at least one PDF."
                )

            else:

                with st.spinner(
                    "Reading and processing your PDFs..."
                ):

                    try:

                        # Extract text
                        raw_text = get_pdf_text(
                            pdf_docs
                        )

                        # Check extracted text
                        if not raw_text.strip():

                            st.error(
                                "No readable text could be extracted from the PDF."
                            )

                            st.info(
                                "The PDF may be scanned/image-based. "
                                "OCR would be required."
                            )

                        else:

                            # --------------------------------
                            # DEBUG EXTRACTION
                            # --------------------------------

                            st.write(
                                f"Extracted {len(raw_text):,} characters."
                            )

                            # --------------------------------
                            # CREATE CHUNKS
                            # --------------------------------

                            text_chunks = get_text_chunks(
                                raw_text
                            )

                            st.write(
                                f"Created {len(text_chunks):,} text chunks."
                            )

                            # --------------------------------
                            # REMOVE OLD DATABASE
                            # --------------------------------

                            if os.path.exists(
                                "faiss_index"
                            ):

                                shutil.rmtree(
                                    "faiss_index"
                                )

                            # --------------------------------
                            # CREATE NEW DATABASE
                            # --------------------------------

                            get_vector_store(
                                text_chunks
                            )

                            st.success(
                                "✅ PDFs processed successfully!"
                            )

                            st.info(
                                "You can now ask questions about your PDFs."
                            )

                    except Exception as e:

                        st.error(
                            f"Error while processing PDF: {e}"
                        )

    # --------------------------------------------------------
    # QUESTION BOX
    # --------------------------------------------------------

    st.markdown("---")

    user_question = st.text_input(
        "💬 Ask a question about your PDF:",
        placeholder="Example: What is this document about?"
    )

    # --------------------------------------------------------
    # ASK QUESTION
    # --------------------------------------------------------

    if user_question:

        if os.path.exists("faiss_index"):

            user_input(
                user_question
            )

        else:

            st.warning(
                "⚠️ Please upload and process your PDF first."
            )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()