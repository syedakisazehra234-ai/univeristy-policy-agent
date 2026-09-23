import os

# ============================================================
# CrewAI/Groq compatibility fix
# ============================================================
#
# Some CrewAI versions inject `cache_breakpoint` into messages.
# Groq rejects that property for its OpenAI-compatible API.
#
# Groq now performs prompt caching automatically, so we do not
# need CrewAI to manually mark cache breakpoints.
#
# This patch must run BEFORE creating the Agent.
# ============================================================

import crewai.llms.cache as _crewai_cache

_crewai_cache.mark_cache_breakpoint = lambda message: message


from crewai import Agent, Crew, LLM, Task

from tools.policy_search import policy_search_tool
from tools.escalation import escalation_tool


# ============================================================
# Environment configuration
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. "
        "Add GROQ_API_KEY to Streamlit Cloud → Settings → Secrets."
    )


GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# ============================================================
# Groq LLM
# ============================================================

llm = LLM(
    model=f"groq/{GROQ_MODEL}",
    api_key=GROQ_API_KEY,
    temperature=0.1,
    max_tokens=2000,
)


# ============================================================
# University Policy Agent
# ============================================================

university_policy_agent = Agent(
    role="University Policy Support Specialist",

    goal=(
        "Help students and university users understand official "
        "university policies, rules, requirements, procedures, "
        "and regulations using the university policy knowledge base."
    ),

    backstory=(
        "You are a university policy support specialist. "
        "Your job is to provide accurate answers based on the "
        "official university policy knowledge base. "
        "You must use the University Policy Search Tool when "
        "answering policy questions. "
        "Never invent university rules, requirements, deadlines, "
        "fees, procedures, or regulations. "
        "If the knowledge base does not contain enough information "
        "to answer the user's question reliably, escalate the "
        "request to human support. "
        "If the user explicitly asks to speak with a human, "
        "escalate the request."
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


# ============================================================
# Public function used by app.py
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

For factual university policy information, use the University
Policy Search Tool.

Do not invent or assume university policies.

If the available policy information is insufficient, escalate
the request to human support.
"""

    else:

        prompt = f"""
Current user question:

{user_message}

Use the University Policy Search Tool to answer the question.

Do not invent university policy information.

If the available policy information is insufficient, escalate
the request to human support.
"""

    task = Task(
        description=prompt,

        expected_output=(
            "A clear, concise and accurate response to the user. "
            "The response must be grounded in the university "
            "policy knowledge base. If the information is "
            "insufficient or the user requests human assistance, "
            "the request must be escalated."
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
