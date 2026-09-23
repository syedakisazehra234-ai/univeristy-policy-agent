import os

import streamlit as st


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="University Policy Assistant",
    page_icon="🎓",
    layout="centered",
)


# =========================================================
# Load Streamlit Secrets into environment variables
# =========================================================

def load_streamlit_secrets():
    """
    Make Streamlit secrets available as environment variables.
    This allows the rest of the project to use os.getenv().
    """

    secret_names = [
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "EMBEDDING_MODEL",
        "TOP_K",
        "MIN_SIMILARITY",
        "MAX_CONTEXT_CHARS",
    ]

    for name in secret_names:
        try:
            value = st.secrets.get(name)

            if value is not None and str(value).strip():
                os.environ[name] = str(value)

        except Exception:
            pass


load_streamlit_secrets()


# Import after environment variables are loaded.
from agent import ask_university_policy


# =========================================================
# Header
# =========================================================

st.title("🎓 University Policy Assistant")

st.write(
    "Ask questions about university policies, rules, "
    "requirements, procedures, and regulations."
)

st.caption(
    "Answers are grounded in the university policy "
    "knowledge base."
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This application uses:

        • CrewAI  
        • A single AI agent  
        • Groq  
        • FAISS semantic search  
        • Sentence Transformers  
        • A pre-built university policy knowledge base
        """
    )

    st.divider()

    st.subheader("Important")

    st.write(
        "The assistant should only treat information "
        "retrieved from the university policy knowledge "
        "base as official policy information."
    )

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# Chat history
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# Example questions
# =========================================================

if not st.session_state.messages:

    st.subheader("Example questions")

    examples = [
        "What is the minimum attendance requirement?",
        "What are the rules for examinations?",
        "What is the procedure for applying for leave?",
        "What are the requirements for graduation?",
    ]

    for example in examples:

        if st.button(
            example,
            use_container_width=True,
        ):
            st.session_state.pending_question = example
            st.rerun()


# =========================================================
# Chat input
# =========================================================

user_question = st.chat_input(
    "Ask a university policy question..."
)


# Support example-question buttons.
if (
    "pending_question" in st.session_state
    and st.session_state.pending_question
):
    user_question = st.session_state.pending_question
    st.session_state.pending_question = None


# =========================================================
# Process question
# =========================================================

if user_question:

    user_question = user_question.strip()

    if not user_question:
        st.stop()

    # Store user message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate answer.
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching university policies..."
        ):

            try:
                conversation_context = "\n".join(
    f"{message['role'].upper()}: {message['content']}"
    for message in st.session_state.messages[-10:]
)

answer = ask_university_policy(
    user_message=user_question,
    conversation_context=conversation_context,
)

            except Exception as exc:

                answer = (
                    "### Application Error\n\n"
                    f"`{type(exc).__name__}: {exc}`\n\n"
                    "Please check the application configuration "
                    "and knowledge-base files."
                )

        st.markdown(answer)

    # Store assistant message.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
