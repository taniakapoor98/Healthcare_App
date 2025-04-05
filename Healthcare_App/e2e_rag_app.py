import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Set page layout for wide screen
st.set_page_config(layout="wide")

# -------------------- Custom CSS Styling --------------------
st.markdown("""
    <style>
    /* Gradient background for the entire app */
    .stApp {
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        color: #FFFFFF;
    }

    /* Card-like background for the chatbot section */
    .chatbot-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Center the Generate Analysis button container */
    .stButton {
        display: flex;
        justify-content: center;
    }

    /* Style only the chat message content container 
       to avoid nested boxes */
    .stChatMessageContent {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        max-width: 90%;
    }

    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.15);
        border: 2px dashed rgba(255, 255, 255, 0.5);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    /* Button styling */
    .stButton > button {
        background: rgba(0, 0, 0, 0.15) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s ease;
        width: auto !important;
        max-width: 200px;
        padding: 10px;
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.25) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    </style>
""", unsafe_allow_html=True)
# -----------------------------------------------------------

# -------------------- Prompts & Models ---------------------
MEDICAL_PROMPT_TEMPLATE = """
You are an expert medical report analyzer specializing in blood test interpretation. 
Analyze the provided blood test report and provide insights in the following format making sure its based on true facts.
Also make sure it is in layman terms, simple and human like as much as possible:

1. Key Findings:
- List the most important observations
- Highlight any values outside normal ranges

2. Risk Assessment:
- Identify potential health risks
- Suggest areas that need attention

3. Recommendations:
- Provide actionable recommendations
- Suggest if any follow-up tests are needed

Query: {user_query}
Report Context: {document_context}
Analysis:
"""

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise, simple, human like and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = './medical_reports/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:latest")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:latest")

# -------------------- Helper Functions ----------------------
def process_medical_report(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
        separators=["\n\n", "\n", ".", ":"]
    )
    return splitter.split_documents(documents)

def clean_analysis(analysis):
    # Remove any <think>...</think> text if present
    end_of_think = analysis.find("</think>")
    if end_of_think != -1:
        return analysis[end_of_think + len("</think>"):].strip()
    return analysis

def analyze_blood_report(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    medical_prompt = ChatPromptTemplate.from_template(MEDICAL_PROMPT_TEMPLATE)
    analysis_chain = medical_prompt | LANGUAGE_MODEL
    return analysis_chain.invoke({"user_query": user_query, "document_context": context_text})

def ans_qry(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    analysis_chain = prompt | LANGUAGE_MODEL
    return analysis_chain.invoke({"user_query": user_query, "document_context": context_text})

# -------------------- Session State Setup -------------------
# This will hold the entire conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# This will hold the Quick Analysis result
if "analysis_report" not in st.session_state:
    st.session_state["analysis_report"] = ""

# -------------------- Layout: 2 Columns (2:1) ---------------
col1, col2 = st.columns([2, 1])

# -------------------- Left Column: Analysis -----------------
with col1:
    st.title("🔬 MedicRep AI")
    st.markdown("### Blood Report Analysis Assistant")
    st.markdown("""
    This AI assistant helps you understand your blood test results by:
    - Analyzing key health indicators
    - Identifying potential risk areas
    - Providing actionable insights
    """)

    # File Upload
    uploaded_report = st.file_uploader(
        "Upload Blood Test Report (PDF)",
        type="pdf",
        help="Upload your blood test report for analysis",
        accept_multiple_files=False
    )

    if uploaded_report:
        saved_path = f"{PDF_STORAGE_PATH}{uploaded_report.name}"
        with open(saved_path, "wb") as file:
            file.write(uploaded_report.getbuffer())

        # Process and index the report
        chunks = process_medical_report(saved_path)
        DOCUMENT_VECTOR_DB.add_documents(chunks)

        st.success("✅ Blood report processed successfully!")

        # Centered "Generate Quick Analysis" Button
        if st.button("Generate Quick Analysis"):
            with st.spinner("Performing comprehensive analysis..."):
                default_query = (
                    "Provide a complete overview of this blood report, "
                    "highlighting any abnormal values and potential health implications."
                )
                relevant_sections = DOCUMENT_VECTOR_DB.similarity_search(default_query)
                analysis = clean_analysis(analyze_blood_report(default_query, relevant_sections))
                # Store in session state so it persists
                st.session_state["analysis_report"] = analysis

    # If we have a saved analysis report, show it
    if st.session_state["analysis_report"]:
        st.markdown("### 🏥 Analysis Report")
        st.write(st.session_state["analysis_report"])

# -------------------- Right Column: Chatbot -----------------
with col2:
    # Wrap the entire chatbot section in a "card"
    st.markdown('<div class="chatbot-card">', unsafe_allow_html=True)
    st.markdown("## 💬 AI Chatbot")
    st.markdown("Ask specific questions about your blood test report.")

    # 1) Display all past messages from session_state
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar=("⚕️" if msg["role"] == "assistant" else None)):
            st.write(msg["content"])

    # 2) Chat input
    user_input = st.chat_input("Ask a question...")

    if user_input:
        # Add the user's message to session state
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Show the user message immediately
        with st.chat_message("user"):
            st.write(user_input)

        # Generate the assistant's reply
        with st.spinner("Analyzing your blood report..."):
            relevant_sections = DOCUMENT_VECTOR_DB.similarity_search(user_input)
            analysis = clean_analysis(ans_qry(user_input, relevant_sections))

        # Add the assistant's reply to session state
        st.session_state["messages"].append({"role": "assistant", "content": analysis})

        # Show the assistant reply immediately
        with st.chat_message("assistant", avatar="⚕️"):
            st.write(analysis)

    st.markdown('</div>', unsafe_allow_html=True)
