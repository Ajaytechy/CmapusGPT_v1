# 🎓 CampusGPT

CampusGPT is an AI-powered University Assistant that helps students instantly find information from university documents, regulations, FAQs, and academic resources through a chatbot-style interface.

Instead of searching through lengthy PDFs, students can simply ask questions in natural language and receive relevant answers with document references.

---

## 🚀 Features

### 👨‍🎓 Student Portal
- Secure student login
- ChatGPT-style interface
- Ask questions in natural language
- Instant answers from university knowledge base
- Source document and page reference display

### 🏫 University Management Portal
- University-specific access
- Upload university documents
- Upload FAQ knowledge bases
- Manage institution-specific information
- Separate knowledge base for each university

### 🤖 AI Capabilities
- Semantic search using Sentence Transformers
- Vector similarity search using FAISS
- Question answering using Hugging Face Transformers
- Context-aware document retrieval
- Source tracking and citation support

---

## 🏗️ Project Architecture

```text
University Documents
        +
FAQ Knowledge Base
        ↓
Sentence Transformer
        ↓
FAISS Vector Search
        ↓
Question Answering Model
        ↓
CampusGPT Response
```

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- PyPDF
- Pandas
- NumPy
- PyTorch

---

## 📂 Project Structure

```text
CampusGPT/
│
├── app.py
├── styles.css
├── requirements.txt
│
├── knowledge_bases/
│
├── data/
│
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/CampusGPT.git
cd CampusGPT
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📚 Example Questions

### Academic

- What is the minimum attendance required?
- How many credits are required for Information Technology?
- What are the examination rules?
- What is the CGPA requirement for distinction?

### Campus Facilities

- How many hostels are available?
- What are the library timings?
- Is transportation available for students?

### Administration

- How do I apply for revaluation?
- What is the scholarship eligibility criteria?
- How can I get my transcript?

---

## 🎯 Problem Statement

Students often struggle to find information from lengthy university documents and regulations.

CampusGPT solves this problem by providing:

- Faster information access
- Reduced administrative workload
- Improved student experience
- AI-powered university support system

---

## 🌟 Future Enhancements

- Multi-university SaaS platform
- Admin dashboard
- Real-time document updates
- Voice-based interaction
- Mobile application
- Role-based access control
- Cloud deployment
- Analytics dashboard

---

## 👨‍💻 Developed By

Ajay

B.Tech Information Technology

CampusGPT – AI Powered University Assistant

---

## 📜 License

This project is developed for educational and hackathon purposes.