# 🎓 University Policy Assistant

A beginner-friendly university policy support AI agent built with:

- CrewAI
- Groq
- FAISS
- Sentence Transformers
- Streamlit

The application uses a pre-built university policy knowledge base rather than uploading or processing the original documents at runtime.

---

## Architecture

```text
User
  ↓
Streamlit
  ↓
CrewAI Single Agent
  ↓
University Policy Search Tool
  ↓
FAISS Vector Database
  +
chunks.json
  ↓
Relevant Policy Context
  ↓
Groq LLM
  ↓
Answer
