import os
import uuid
import base64
import dotenv
from PIL import Image
import io
import pytesseract
import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
import openai

dotenv.load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(layout="wide")

# -------------------- Custom CSS Styling --------------------
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%); color: #FFFFFF; }
    .chatbot-card { background: rgba(255, 255, 255, 0.15); border-radius: 15px; padding: 20px; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .stButton { display: flex; justify-content: center; }
    .stChatMessageContent { background: rgba(255, 255, 255, 0.1) !important; border-radius: 12px; padding: 12px; margin: 8px 0; border: 1px solid rgba(255, 255, 255, 0.3); backdrop-filter: blur(10px); max-width: 90%; }
    .stFileUploader { background: rgba(255, 255, 255, 0.15); border: 2px dashed rgba(255, 255, 255, 0.5); border-radius: 10px; padding: 20px; text-align: center; backdrop-filter: blur(10px); }
    .stButton > button { background: rgba(0, 0, 0, 0.15) !important; color: #FFFFFF !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; transition: all 0.3s ease; width: auto !important; max-width: 200px; padding: 10px; }
    .stButton > button:hover { background: rgba(255, 255, 255, 0.25) !important; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); }
    .image-preview-container { 
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
    .image-preview-box {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 10px;
        width: 150px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        backdrop-filter: blur(5px);
        transition: transform 0.3s ease;
        overflow: hidden;
    }
    .image-preview-box:hover {
        transform: scale(1.05);
    }
    .image-preview-box img {
        width: 100%;
        height: 100px;
        object-fit: cover;
        border-radius: 6px;
        margin-bottom: 5px;
    }
    .image-preview-box p {
        font-size: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin: 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)
# -----------------------------------------------------------

# -------------------- Prompts & Models ---------------------
MEDICAL_PROMPT_TEMPLATE = """
You are an expert medical analyzer specializing in blood test and healthcare related images interpretation. 
Analyze the provided blood test report and images and provide insights in the following format making sure its based on true facts.
Also make sure it is in layman terms, simple and human like as much as possible:

1. Key Findings:
- List the most important observations based on the image summaries
- List the most important observations based on the blood reports
- Highlight any values outside normal ranges

2. Risk Assessment:
- Identify potential health risks
- Suggest areas that need attention

3. Recommendations:
- Provide actionable recommendations
- Suggest if any follow-up tests are needed

At the end of your analysis, add a new section:
4. Missing Information:
- List any key details that are missing but would be helpful for a more accurate analysis. 
- If no details are missing, output "NO_MISSING_DETAILS"

Query: {user_query}
Report Context: {document_context}
Analysis:
"""

PROMPT_TEMPLATE = """
You are a healthcare expert. Use the provided context of images and PDFs to answer the query. 
If unsure, state that you don't know and would need more information. Be concise, simple, human like and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""


# -------------------- Helper Functions ----------------------
def process_image(image):
    text = pytesseract.image_to_string(image)
    return text


def encode_image(uploaded_file):
    file_content = uploaded_file.read()
    return base64.b64encode(file_content).decode('utf-8')


def get_image_object(uploaded_file):
    # Return to the beginning of the file after reading
    uploaded_file.seek(0)
    return Image.open(uploaded_file)


def summarize_image(encoded_image):
    prompt = [
        SystemMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[{
            "type": "text",
            "text": "Describe the contents of this image."
        },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, max_tokens=1024).invoke(prompt)
    return response.content


def count_tokens(text):
    return len(text.split())


def extract_missing_details(analysis_text):
    sections = analysis_text.split("4. Missing Information:")
    if len(sections) > 1:
        missing_info = sections[1].strip()
        if "NO_MISSING_DETAILS" in missing_info:
            return None
        else:
            # Extract the list of missing details
            missing_details = [item.strip() for item in missing_info.split("\n") if item.strip().startswith("-")]
            return missing_details
    return None


# -------------------- Session State Setup -------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "analysis_report" not in st.session_state:
    st.session_state["analysis_report"] = ""
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False
if "missing_details" not in st.session_state:
    st.session_state["missing_details"] = None
if "uploaded_image_objects" not in st.session_state:
    st.session_state["uploaded_image_objects"] = []

# -------------------- Layout: 2 Columns (2:1) ---------------
col1, col2 = st.columns([2, 1])

# -------------------- Left Column: Analysis -----------------
with col1:
    st.title("🔬 MedicRep AI")
    st.markdown("### Health Report Analysis Assistant")
    st.markdown("""
    This AI assistant helps you understand your health reports by:
    - Analyzing key health indicators
    - Identifying potential risk areas
    - Providing actionable insights
    """)

    # File Upload for PDFs
    uploaded_report = st.file_uploader("Upload Health Report (PDF)", type="pdf",
                                       help="Upload your health report for analysis", accept_multiple_files=False)

    # File Upload for Images
    uploaded_images = st.file_uploader("Upload Images (JPG, PNG)", type=["jpg", "png", "jpeg"],
                                       help="Upload images related to your health report", accept_multiple_files=True)

    # Display image previews in cute boxes
    if uploaded_images:
        st.markdown("### 📸 Uploaded Images")

        # Store image objects if they're not already stored
        current_image_names = [img.name for img in uploaded_images]
        stored_image_names = [img.name if hasattr(img, 'name') else '' for img in
                              st.session_state["uploaded_image_objects"]]

        # Reset image objects if the selection has changed
        if set(current_image_names) != set(stored_image_names):
            st.session_state["uploaded_image_objects"] = []
            for uploaded_image in uploaded_images:
                try:
                    image = get_image_object(uploaded_image)
                    # Store the image object along with its name
                    image.name = uploaded_image.name
                    st.session_state["uploaded_image_objects"].append(image)
                except Exception as e:
                    st.error(f"Error loading image {uploaded_image.name}: {e}")

        # Create a container for the preview boxes
        st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)

        # Display preview boxes
        for idx, img in enumerate(st.session_state["uploaded_image_objects"]):
            # Convert PIL image to bytes for display
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            # Create HTML for each preview box
            image_name = img.name if hasattr(img, 'name') else f"Image {idx + 1}"
            col_html = f"""
            <div class="image-preview-box">
                <img src="data:image/jpeg;base64,{base64.b64encode(byte_im).decode()}" alt="{image_name}">
                <p>{image_name}</p>
            </div>
            """
            st.markdown(col_html, unsafe_allow_html=True)

        # Close the container
        st.markdown('</div>', unsafe_allow_html=True)

    text_elements = []
    text_summaries = []
    image_elements = []
    image_summaries = []

    if uploaded_report and uploaded_report.name.lower().endswith('.pdf'):
        with open("temp_report.pdf", "wb") as f:
            f.write(uploaded_report.getbuffer())
        # Extract text from the PDF
        loader = PDFPlumberLoader("temp_report.pdf")
        pdf_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pdf_docs)

        for doc in docs:
            text_elements.append(doc.page_content)
            text_summaries.append(doc.page_content)

    if uploaded_images:
        for uploaded_image in uploaded_images:
            # Reset file pointer to the beginning before reading
            uploaded_image.seek(0)
            encoded_image = encode_image(uploaded_image)
            image_elements.append(encoded_image)
            summary = summarize_image(encoded_image)
            image_summaries.append(summary)

    documents = []
    for e, s in zip(text_elements, text_summaries):
        i = str(uuid.uuid4())
        doc = Document(page_content=s, metadata={'id': i, 'type': 'text', 'original_content': e})
        documents.append(doc)

    for e, s in zip(image_elements, image_summaries):
        i = str(uuid.uuid4())
        doc = Document(page_content=s, metadata={'id': i, 'type': 'image', 'original_content': s})
        documents.append(doc)

    if documents:
        vectorstore = FAISS.from_documents(documents=documents,
                                           embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))
        vectorstore.save_local("faiss_index")
        st.success("✅ Health report and images processed successfully!")

    symptoms = st.text_input("Enter your symptoms")

    # Analyze button
    if st.button("Analyze"):
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        medical_prompt = ChatPromptTemplate.from_template(MEDICAL_PROMPT_TEMPLATE) | ChatOpenAI(model="gpt-4o",
                                                                                                openai_api_key=openai_api_key,
                                                                                                max_tokens=1024)


        def analyze_medical_report():
            relevant_docs = db.similarity_search("Provide a complete overview of this health report.")
            context = "\n\n".join([d.metadata['original_content'] for d in relevant_docs])
            token_count = count_tokens(context)
            if token_count > 10000:
                st.error("The context is too large. Please reduce the input size.")
                return "Context too large."
            result = medical_prompt.invoke({'user_query': symptoms, 'document_context': context})
            return result


        complete_result = analyze_medical_report()
        # Split the result to remove the "Missing Information" section for display
        display_result = complete_result.content.split("4. Missing Information:")[0] if hasattr(complete_result,
                                                                                                'content') else complete_result

        # Store the missing details in session state
        if hasattr(complete_result, 'content'):
            st.session_state["missing_details"] = extract_missing_details(complete_result.content)
        else:
            st.session_state["missing_details"] = extract_missing_details(complete_result)

        st.write("Analysis:", display_result)
        st.session_state["analysis_complete"] = True

        # Auto-add a message to the chatbot based on missing details
        if st.session_state["analysis_complete"] and len(st.session_state["messages"]) == 0:
            if st.session_state["missing_details"]:
                missing_info_message = "I noticed some key details are missing from your report. Could you please provide more information about: " + ", ".join(
                    [item.replace("- ", "") for item in st.session_state["missing_details"]])
                st.session_state["messages"].append({"role": "assistant", "content": missing_info_message})
            else:
                st.session_state["messages"].append({"role": "assistant", "content": "How are you feeling today?"})

# -------------------- Right Column: Chatbot -----------------
with col2:
    st.markdown('<div class="chatbot-card">', unsafe_allow_html=True)
    st.markdown("## 💬 AI Chatbot")
    st.markdown("Ask specific questions about your health report.")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar=("⚕️" if msg["role"] == "assistant" else None)):
            st.write(msg["content"])

    # Move the chat input to the end
    user_input = st.chat_input("Ask a question...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.spinner("Analyzing your health report..."):
            qa_chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | ChatOpenAI(model="gpt-4o",
                                                                                      openai_api_key=openai_api_key,
                                                                                      max_tokens=1024)


            def answer_question(question):
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

                relevant_docs = db.similarity_search(question)
                context = "\n\n".join([d.metadata['original_content'] for d in relevant_docs])
                token_count = count_tokens(context)
                if token_count > 10000:
                    st.error("The context is too large. Please reduce the input size.")
                    return "Context too large."
                result = qa_chain.invoke({'user_query': question, 'document_context': context})
                return result


            result = answer_question(user_input)

        st.session_state["messages"].append(
            {"role": "assistant", "content": result.content if hasattr(result, 'content') else result})
        with st.chat_message("assistant", avatar="⚕️"):
            st.write(result.content if hasattr(result, 'content') else result)

    st.markdown('</div>', unsafe_allow_html=True)