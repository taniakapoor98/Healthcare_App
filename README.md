# MedicRep AI 🧬🩺 | Your AI Healthcare Companion

[![Watch the video](https://img.youtube.com/vi/Gyd72iFg2Ek/0.jpg)](https://www.youtube.com/watch?v=Gyd72iFg2Ek)

MedicRep AI is your intelligent healthcare assistant that helps you analyze blood test reports and symptoms using state-of-the-art multimodal AI. Empower patients with fast, accurate, and interpretable health insights.

---

## 🧠 Features

- 📄 Blood Report Upload & Analysis (PDF parsing + structured data)
- 🖼️ Symptom Image Support (AI-based interpretation)
- 🧍‍♂️ Patient-described Symptoms (cross-referenced with medical literature)
- 🧾 Simple, clear health summaries + Recommendations
- 🧠 Multimodal RAG for context-aware memory + retrieval
- 💬 Chatbot interface to ask questions about results
- ⚠️ Confidence-aware responses to ensure safety and reliability

---

## 🏗️ Architecture

```
[User Inputs: PDF, Image, Text]
        |
        v
[Preprocessing Layer: OCR, Parsing, Embeddings]
        |
        v
[Multimodal RAG Engine]
        |
        v
[LLM with Health Knowledge]
        |
        v
[Output: Summary + Explanation + Suggestions]
```

---

## 🧪 Tech Stack

- LangChain 🦜🔗
- Azure OpenAI GPT-4
- Multimodal Retrieval-Augmented Generation (RAG)
- FastAPI + React for front-end/backend
- Azure Blob Storage for file handling
- Tesseract OCR, PyMuPDF for PDF parsing
- Faiss + Hybrid Search for embeddings

---

## 📈 Example Output

- 🩸 Uploaded a blood report PDF → Shows flagged values + interpretation (e.g., low hemoglobin = anemia risk)
- 💬 Described symptoms like fatigue + chest pain → Returns possible causes with confidence score
- 📷 Uploaded skin rash image → AI highlights potential causes (eczema, allergic reaction, etc.)

---

## 💡 Why MedicRep AI?

Patients struggle to understand medical data. This tool bridges the gap using AI—demystifying lab reports, symptom images, and vague feelings into actionable insights.

---

## 📬 Contribute / Collaborate

PRs welcome! Reach out on [LinkedIn](https://www.linkedin.com/in/tania-kapoor-0450b0188/).

---
Made with ❤️ for better patient understanding.


