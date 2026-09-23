```python
import os

from crewai import Agent, LLM

from tools.policy_search import policy_search_tool
from tools.escalation import escalation_tool


# ============================================================
# Configuration
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. "
        "Add it to Streamlit Cloud → Settings → Secrets."
    )


# Current Groq model used by this project.
# gpt-oss-120b is available on Groq.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# ============================================================
# Groq LLM
# ============================================================
#
# IMPORTANT:
# Do not pass cache_breakpoint or other provider-specific
# caching parameters through CrewAI/LiteLLM.
#
# Groq's API does not accept cache_breakpoint in a system
# message.
#
# We keep the LLM configuration deliberately minimal.
# ============================================================

llm = LLM(
    model=f"groq/{GROQ_MODEL}",
    api_key=GROQ_API_KEY,
    temperature=0.1,
    max_tokens=2000,
)


# ============================================================
# University Policy Support Agent
# ============================================================

university_policy_agent = Agent(
    role="University Policy Support Specialist",

    goal=(
        "Help users understand university policies using only "
        "the information available in the university policy "
        "knowledge base. Provide accurate, clear and useful "
        "answers and escalate issues to human support when "
        "necessary."
    ),

    backstory=(
        "You are a university policy support specialist. "
        "You help students and university users understand "
        "official university policies, regulations and "
        "procedures. You must use the University Policy Search "
        "Tool before answering policy questions. You must never "
        "invent university rules, requirements, deadlines or "
        "procedures. If the knowledge base does not contain "
        "enough information to answer the question reliably, "
        "you must escalate the issue to human support. "
        "If the user explicitly asks to speak with a human, "
        "you must escalate the request immediately."
    ),

    tools=[
        policy_search_tool,
        escalation_tool,
    ],

    llm=llm,

    verbose=False,

    allow_delegation=False,

    max_iter=5,

    memory=False,
)
```

### Why `memory=False` here?

This does **not** remove your chat memory.

Your Streamlit conversation memory should remain in:

```python
st.session_state.messages
```

That's actually preferable for this application.

We're separating:

```text
Conversation memory
        ↓
Streamlit session_state

Knowledge
        ↓
FAISS + chunks.json

Agent
        ↓
CrewAI

LLM
        ↓
Groq
```

That prevents CrewAI's internal memory/caching machinery from interfering with the Groq request.

---

# 2. Also change your `requirements.txt`

Your previous setup had fairly aggressive package versions. I would simplify the dependency file.

Use:

```text
streamlit>=1.50,<2.0
crewai>=1.15,<2.0
crewai-tools>=1.15,<2.0

groq>=0.33,<1.0
litellm>=1.89,<2.0

faiss-cpu>=1.12,<2.0
sentence-transformers>=5.1,<7.0
numpy>=2.0,<3.0
```

The important point isn't simply installing the newest possible package. It's avoiding an old LiteLLM/CrewAI combination that may inject provider parameters your Groq endpoint doesn't accept.

LiteLLM maps provider-specific 400 errors into its own `BadRequestError`, which is why your Streamlit traceback says `litellm.BadRequestError` even though the underlying rejection is coming from Groq.

---

# 3. Check your `app.py`

There is another thing I want you to check.

You should **not** have anything like this:

```python
LLM(
    model="groq/openai/gpt-oss-120b",
    cache=True,
    cache_breakpoint=True,
)
```

or:

```python
LLM(
    model="groq/openai/gpt-oss-120b",
    additional_params={
        "cache_breakpoint": ...
    }
)
```

or:

```python
llm_config = {
    "cache_breakpoint": ...
}
```

Search your entire repository for:

```text
cache_breakpoint
```

If it appears anywhere in your own code, **remove it**.

---

# 4. There is another important issue with your model

Your current project is using:

```text
openai/gpt-oss-120b
```

That's fine if that is the model you selected in Groq.

However, don't use the old:

```text
llama-3.3-70b-versatile
```

configuration from your earlier project.

Groq's current documentation says `llama-3.3-70b-versatile` was deprecated and shut down on **August 16, 2026**, with `openai/gpt-oss-120b` listed as one of the replacements.

So your current:

```text
openai/gpt-oss-120b
```

choice is appropriate for the current date.

---

# 5. Your Streamlit Secrets

In Streamlit Cloud → **Settings → Secrets**, keep:

```toml
GROQ_API_KEY = "your-groq-api-key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Do **not** put:

```toml
cache_breakpoint = true
```

or any other cache configuration there.

---

# 6. Your FAISS files do NOT need to be rebuilt

This error:

```text
GroqException
```

happens at the LLM request stage.

Your pipeline is effectively:

```text
Streamlit
   ↓
CrewAI
   ↓
LiteLLM
   ↓
Groq
   ↓
❌ cache_breakpoint rejected
```

It has nothing to do with:

```text
faiss.index
chunks.json
```

So **don't regenerate your embeddings just because of this error**.

Your existing:

```text
knowledge_base/
├── faiss.index
└── chunks.json
```

can remain unchanged.

---

# 7. One more fix: make the policy tool independent of CrewAI caching

Your `policy_search.py` should only do:

```text
question
   ↓
SentenceTransformer
   ↓
FAISS
   ↓
chunks.json
   ↓
retrieved context
```

It should **not** call Gemini/Groq itself.

The architecture should remain:

```text
                  USER
                    │
                    ▼
              Streamlit
                    │
                    ▼
                CrewAI
                    │
                    ▼
          University Agent
             /          \
            /            \
           ▼              ▼
      Policy Search     Escalation
         Tool             Tool
           │                │
           ▼                ▼
       FAISS             Pending
       chunks             request
           │
           ▼
        Context
           │
           ▼
          Groq
           │
           ▼
         Answer
```

---

# 8. Also check the import

Your previous application error showed:

```python
from agent import ask_university_policy
```

If your current `app.py` still uses:

```python
from agent import ask_university_policy
```

then your `agent.py` needs to actually expose that function.

If my previous architecture used that function, don't simply replace the entire file with an `Agent` object and leave `app.py` unchanged.

Instead, use this complete pattern:

```python
import os

from crewai import Agent, Crew, LLM, Task

from tools.policy_search import policy_search_tool
from tools.escalation import escalation_tool


# ============================================================
# Environment
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured in Streamlit Secrets."
    )

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# ============================================================
# LLM
# ============================================================

llm = LLM(
    model=f"groq/{GROQ_MODEL}",
    api_key=GROQ_API_KEY,
    temperature=0.1,
    max_tokens=2000,
)


# ============================================================
# Agent
# ============================================================

university_policy_agent = Agent(
    role="University Policy Support Specialist",

    goal=(
        "Answer university policy questions accurately using "
        "the university policy knowledge base."
    ),

    backstory=(
        "You are a university policy support specialist. "
        "Always use the policy search tool for policy questions. "
        "Never invent university rules or procedures. "
        "If the available knowledge base does not contain "
        "enough information, escalate the issue to human "
        "support. If the user explicitly asks for a human, "
        "escalate the request."
    ),

    tools=[
        policy_search_tool,
        escalation_tool,
    ],

    llm=llm,

    allow_delegation=False,

    verbose=False,

    memory=False,

    max_iter=5,
)


# ============================================================
# Public function used by Streamlit
# ============================================================

def ask_university_policy(
    user_message: str,
    conversation_context: str = "",
) -> str:

    context_text = conversation_context.strip()

    if context_text:
        prompt = f"""
Previous conversation:

{context_text}

Current user question:

{user_message}

Use the previous conversation only as conversational context.
Use the University Policy Search Tool for factual policy
information.

Answer the current question clearly.
"""
    else:
        prompt = f"""
User question:

{user_message}

Use the University Policy Search Tool to answer the question.
Do not invent university policy information.
"""

    task = Task(
        description=prompt,

        expected_output=(
            "A clear and accurate response to the user's question. "
            "If the policy knowledge base is insufficient or the "
            "user requests human assistance, escalate the request."
        ),

        agent=university_policy_agent,
    )

    crew = Crew(
        agents=[university_policy_agent],
        tasks=[task],
        verbose=False,
    )

    result = crew.kickoff()

    return str(result)
```

This preserves the interface:

```python
from agent import ask_university_policy
```

that your existing Streamlit application appears to expect.

---

# 9. Why this should fix the specific error

The problematic request currently looks conceptually like:

```json
{
  "role": "system",
  "content": "...",
  "cache_breakpoint": true
}
```

Groq rejects it.

The corrected configuration sends an ordinary system message without that unsupported field:

```json
{
  "role": "system",
  "content": "You are a university policy support specialist..."
}
```

Groq's documented Chat API accepts `messages` as the conversation input and does not list `cache_breakpoint` among the supported message properties.

Interestingly, Groq **does support automatic prompt caching**, but its current documentation describes caching as automatic/exact-prefix based rather than requiring you to manually inject `cache_breakpoint` into a system message.

---

# 10. What to do on GitHub now

Make these changes:

```text
1. agent.py
   ↓
   replace with corrected version

2. requirements.txt
   ↓
   replace with corrected versions

3. Search repository
   ↓
   cache_breakpoint

4. If found
   ↓
   remove it

5. Commit
   ↓
   Push to GitHub

6. Streamlit Cloud
   ↓
   Reboot app
```

You **do not need to touch**:

```text
rag/faiss.index
rag/chunks.json
```

unless the application subsequently reports an actual FAISS/chunks loading error.

### One caveat

If the error **still says `cache_breakpoint` after these changes**, then the parameter is almost certainly being injected by the installed CrewAI/LiteLLM version rather than your application code. In that case, the next thing to inspect is the exact `requirements.txt` and the current `agent.py` from your repository; don't rebuild the knowledge base. Groq's current API itself is rejecting the unsupported field, so the fix belongs at the integration layer.

If you paste/upload your **current `app.py`, `agent.py`, `tools/policy_search.py`, `tools/escalation.py`, and `requirements.txt`**, I can check the actual code path and give you the corrected versions rather than guessing at which layer is injecting `cache_breakpoint`.

