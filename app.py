import re
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CampusGPT",
    page_icon="🎓",
    layout="wide"
)

def load_css():
    css_path = Path(__file__).parent / "styles.css"

    with open(css_path, "r", encoding="utf-8") as file:
        css = file.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )


load_css()

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "knowledge_bases"


# =========================================================
# UNIVERSITIES
# =========================================================

UNIVERSITIES = {
    "university_a": {
        "name": "Bannari Amman Institute of Technology"
    },
    "university_b": {
        "name": "Demo University"
    }
}


# =========================================================
# USERS
# =========================================================

USERS = {

    "student_a": {
        "password": "1234",
        "role": "student",
        "university_id": "university_a",
        "name": "Student A"
    },

    "student_b": {
        "password": "1234",
        "role": "student",
        "university_id": "university_b",
        "name": "Student B"
    },

    "admin_a": {
        "password": "1234",
        "role": "admin",
        "university_id": "university_a",
        "name": "University A Admin"
    },

    "admin_b": {
        "password": "1234",
        "role": "admin",
        "university_id": "university_b",
        "name": "University B Admin"
    }
}


# =========================================================
# CREATE REQUIRED FOLDERS
# =========================================================

for university_id in UNIVERSITIES:

    document_folder = (
        DATA_DIR
        / university_id
        / "documents"
    )

    knowledge_folder = (
        KB_DIR
        / university_id
    )

    document_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    knowledge_folder.mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# LOAD QUESTION ANSWERING MODEL
# =========================================================

@st.cache_resource
def load_qa_model():

    model_name = "distilbert-base-cased-distilled-squad"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name
    )

    model.eval()

    return tokenizer, model

tokenizer, qa_model = load_qa_model()
# =========================================================
# ANSWER EXTRACTION
# =========================================================

def get_model_answer(question, context):

    tokenizer, qa_model = load_qa_model()

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
        return_offsets_mapping=True
    )

    offset_mapping = inputs["offset_mapping"]

    model_inputs = {
        key: value
        for key, value in inputs.items()
        if key != "offset_mapping"
    }

    with torch.no_grad():

        outputs = qa_model(
            **model_inputs
        )

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    sequence_ids = inputs.sequence_ids(0)

    for i, sequence_id in enumerate(sequence_ids):

        if sequence_id != 1:

            start_logits[i] = -float("inf")
            end_logits[i] = -float("inf")

    start_index = int(
        torch.argmax(start_logits)
    )

    best_end_index = start_index
    best_score = -float("inf")

    max_answer_length = 15

    for end_index in range(
        start_index,
        min(
            start_index + max_answer_length,
            len(end_logits)
        )
    ):

        score = (
            start_logits[start_index].item()
            + end_logits[end_index].item()
        )

        if score > best_score:

            best_score = score
            best_end_index = end_index

    start_char = (
        offset_mapping[0][start_index][0].item()
    )

    end_char = (
        offset_mapping[0][best_end_index][1].item()
    )

    return context[
        start_char:end_char
    ].strip()


# =========================================================
# ATTENDANCE ANSWER REFINEMENT
# =========================================================

def refine_answer(question, context, model_answer):

    q = question.lower()

    if "attendance" in q:

        match = re.search(
            r"not less than\s+(\d+)%\s+overall attendance",
            context,
            re.IGNORECASE
        )

        if match:

            return (
                f"The minimum overall attendance "
                f"required is {match.group(1)}%."
            )

        match = re.search(
            r"minimum.*?(\d+)%.*?attendance",
            context,
            re.IGNORECASE
        )

        if match:

            return (
                f"The minimum overall attendance "
                f"required is {match.group(1)}%."
            )

    return model_answer


# =========================================================
# EXTRACT PDF CHUNKS
# =========================================================

def extract_pdf_chunks(pdf_path):

    reader = PdfReader(pdf_path)

    chunks = []
    metadata = []

    chunk_size = 80
    overlap = 20

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if not page_text:
            continue

        words = page_text.split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk = " ".join(
                words[start:end]
            ).strip()

            if chunk:

                chunks.append(chunk)

                metadata.append(
                    {
                        "file": pdf_path.name,
                        "page": page_number
                    }
                )

            start += chunk_size - overlap

    return chunks, metadata


# =========================================================
# BUILD + SAVE UNIVERSITY KNOWLEDGE BASE
# =========================================================

def build_and_save_knowledge_base(university_id):

    document_folder = (
        DATA_DIR
        / university_id
        / "documents"
    )

    knowledge_folder = (
        KB_DIR
        / university_id
    )

    pdf_files = sorted(
        document_folder.glob("*.pdf")
    )

    if not pdf_files:

        return False, 0

    all_chunks = []
    all_metadata = []

    # -----------------------------------------------------
    # READ ALL UNIVERSITY PDFs
    # -----------------------------------------------------

    for pdf_file in pdf_files:

        chunks, metadata = extract_pdf_chunks(
            pdf_file
        )

        all_chunks.extend(chunks)
        all_metadata.extend(metadata)

    if not all_chunks:

        return False, 0

    # -----------------------------------------------------
    # CREATE EMBEDDINGS
    # -----------------------------------------------------

    embeddings = embedding_model.encode(
        all_chunks,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(
        embeddings
    )

    # -----------------------------------------------------
    # CREATE FAISS INDEX
    # -----------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    # -----------------------------------------------------
    # SAVE INDEX
    # -----------------------------------------------------

    index_path = (
        knowledge_folder
        / "index.faiss"
    )

    metadata_path = (
        knowledge_folder
        / "metadata.pkl"
    )

    chunks_path = (
        knowledge_folder
        / "chunks.pkl"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    with open(
        metadata_path,
        "wb"
    ) as file:

        pickle.dump(
            all_metadata,
            file
        )

    with open(
        chunks_path,
        "wb"
    ) as file:

        pickle.dump(
            all_chunks,
            file
        )

    return True, len(all_chunks)


# =========================================================
# LOAD SAVED KNOWLEDGE BASE
# =========================================================

def load_knowledge_base(university_id):

    knowledge_folder = (
        KB_DIR
        / university_id
    )

    index_path = (
        knowledge_folder
        / "index.faiss"
    )

    metadata_path = (
        knowledge_folder
        / "metadata.pkl"
    )

    chunks_path = (
        knowledge_folder
        / "chunks.pkl"
    )

    if not (
        index_path.exists()
        and metadata_path.exists()
        and chunks_path.exists()
    ):

        return None, [], []

    index = faiss.read_index(
        str(index_path)
    )

    with open(
        metadata_path,
        "rb"
    ) as file:

        metadata = pickle.load(
            file
        )

    with open(
        chunks_path,
        "rb"
    ) as file:

        chunks = pickle.load(
            file
        )

    return (
        index,
        chunks,
        metadata
    )


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False

if "current_user" not in st.session_state:

    st.session_state.current_user = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================================
# LOGIN
# =========================================================

if not st.session_state.logged_in:

    st.title("🎓 CampusGPT")

    st.subheader(
        "Multi-University AI Knowledge Assistant"
    )

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        use_container_width=True
    ):

        if username in USERS:

            user = USERS[username]

            if password == user["password"]:

                st.session_state.logged_in = True
                st.session_state.current_user = user

                st.rerun()

            else:

                st.error(
                    "Incorrect password."
                )

        else:

            st.error(
                "User not found."
            )

    st.info(
        "Demo Accounts: "
        "student_a / 1234 | "
        "student_b / 1234 | "
        "admin_a / 1234 | "
        "admin_b / 1234"
    )


# =========================================================
# LOGGED-IN AREA
# =========================================================

else:

    user = st.session_state.current_user

    university_id = user["university_id"]

    university_name = (
        UNIVERSITIES[
            university_id
        ]["name"]
    )

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.title(
        "🎓 CampusGPT"
    )

    st.sidebar.write(
        f"**User:** {user['name']}"
    )

    st.sidebar.write(
        f"**Institution:** {university_name}"
    )

    st.sidebar.write(
        f"**Role:** {user['role'].title()}"
    )

    if st.sidebar.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.current_user = None

        st.rerun()


    # =====================================================
    # ADMIN PORTAL
    # =====================================================

    if user["role"] == "admin":

        st.title(
            "🏫 University Admin Dashboard"
        )

        st.subheader(
            university_name
        )

        st.write(
            "Upload official documents and update "
            "your university's AI knowledge base."
        )

        # -------------------------------------------------
        # UPLOAD DOCUMENTS
        # -------------------------------------------------

        uploaded_files = st.file_uploader(
            "📄 Upload Institutional PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:

            document_folder = (
                DATA_DIR
                / university_id
                / "documents"
            )

            for uploaded_file in uploaded_files:

                file_path = (
                    document_folder
                    / uploaded_file.name
                )

                with open(
                    file_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

            st.success(
                f"{len(uploaded_files)} "
                f"document(s) uploaded successfully."
            )

            # -------------------------------------------------
            # AUTOMATIC KNOWLEDGE BASE UPDATE
            # -------------------------------------------------

            with st.spinner(
                "Processing documents and building AI knowledge base..."
            ):

                success, chunk_count = (
                    build_and_save_knowledge_base(
                        university_id
                    )
                )

            if success:

                st.success(
                    f"✅ Knowledge base updated successfully. "
                    f"{chunk_count} searchable sections created."
                )

            else:

                st.error(
                    "Unable to build the knowledge base."
                )

        # -------------------------------------------------
        # DOCUMENT LIBRARY
        # -------------------------------------------------

        st.markdown(
            "### 📚 Document Library"
        )

        document_folder = (
            DATA_DIR
            / university_id
            / "documents"
        )

        documents = sorted(
            document_folder.glob("*.pdf")
        )

        if documents:

            for document in documents:

                st.write(
                    f"📄 {document.name}"
                )

        else:

            st.info(
                "No documents uploaded yet."
            )


    # =====================================================
    # STUDENT PORTAL
    # =====================================================

    else:

        st.markdown(
         """
            <div class="campus-title">
              CampusGPT
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="campus-subtitle">
                {university_name} · Academic Knowledge Assistant
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-title">
                How can I help you?
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Ask questions about attendance, examinations, fees, regulations, and other institutional documents."
        )

    
        question = st.chat_input(
            "Ask CampusGPT about your university..."
        )

        if question:

            # -------------------------------------------------
            # LOAD UNIVERSITY-SPECIFIC KNOWLEDGE BASE
            # -------------------------------------------------

            index, chunks, metadata = (
                load_knowledge_base(
                    university_id
                )
            )

            if index is None:

                st.warning(
                    "Your university's knowledge base "
                    "is not available yet."
                )

            else:

                # -------------------------------------------------
                # QUESTION EMBEDDING
                # -------------------------------------------------

                question_embedding = (
                    embedding_model.encode(
                        [question],
                        convert_to_numpy=True
                    ).astype("float32")
                )

                faiss.normalize_L2(
                    question_embedding
                )

                                # -------------------------------------------------
                # SEARCH TOP 5 RELEVANT CHUNKS
                # -------------------------------------------------

                scores, results = index.search(
                    question_embedding,
                    5
                )

                best_index = int(results[0][0])
                best_context = chunks[best_index]
                source = metadata[best_index]
                st.write(best_context)

                # -------------------------------------------------
                # GET ANSWER
                # -------------------------------------------------

                try:

                    model_answer = get_model_answer(
                        question,
                        best_context
                    )

                except Exception:

                    model_answer = best_context

                # -------------------------------------------------
                # REFINE ANSWER
                # -------------------------------------------------

                answer = refine_answer(
                    question,
                    best_context,
                    model_answer
                )

                # -------------------------------------------------
                # SAVE CHAT
                # -------------------------------------------------

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "source": source["file"],
                        "page": source["page"]
                    }
                )

                # -------------------------------------------------
                # DISPLAY CHAT
                # -------------------------------------------------

                for chat in st.session_state.chat_history:

                    st.markdown(
                        f"""
                        <div class="user-message">
                            <div class="message-title">You</div>
                            <div>{chat["question"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="bot-message">
                            <div class="message-title">CampusGPT</div>
                            <div>{chat["answer"]}</div>
                            <div class="message-source">
                                📄 {chat["source"]} — Page {chat["page"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )