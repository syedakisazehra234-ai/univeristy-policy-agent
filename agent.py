import os

from crewai import Agent, Crew, LLM, Process, Task

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    validate_configuration,
)

from rag_tool import university_policy_search


# =========================================================
# LLM
# =========================================================

def create_llm() -> LLM:
    """
    Create the Groq LLM used by CrewAI.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    return LLM(
        model=f"groq/{GROQ_MODEL}",
        api_key=GROQ_API_KEY,
        temperature=0.1,
    )


# =========================================================
# Agent
# =========================================================

def create_agent() -> Agent:
    """
    Create the single university policy support agent.
    """

    llm = create_llm()

    return Agent(
        role="University Policy Support Specialist",

        goal=(
            "Answer university policy questions accurately "
            "using the university policy knowledge base. "
            "Never invent institutional rules."
        ),

        backstory=(
            "You are a careful university policy support "
            "specialist. You help students and staff understand "
            "institutional policies in clear and practical "
            "language. You always search the provided policy "
            "knowledge base before answering policy questions. "
            "You distinguish official policy information from "
            "general knowledge and clearly state when the "
            "available policy information is insufficient."
        ),

        tools=[
            university_policy_search
        ],

        llm=llm,

        verbose=False,

        allow_delegation=False,

        max_iter=4,
    )


# =========================================================
# Task
# =========================================================

def create_task(agent: Agent, question: str) -> Task:
    """
    Create the task for the single agent.
    """

    return Task(
        description=f"""
Answer the following university policy question:

{question}

Follow these rules carefully:

1. Use the university_policy_search tool before answering
   any question about university policy.

2. Base policy claims on the retrieved knowledge-base content.

3. Do not invent policies, requirements, deadlines, fees,
   procedures, exceptions, offices, contact information,
   document names, URLs, or page numbers.

4. Do not use general knowledge as if it were official
   university policy.

5. If the retrieved information is insufficient, clearly say
   that the available policy knowledge base does not provide
   enough information.

6. If the retrieved sources contain an exception or special
   condition, preserve it in the answer.

7. If multiple retrieved sources appear to conflict, mention
   the conflict rather than silently choosing one.

8. Answer the user's question directly first.

9. Use concise language, but include important conditions,
   exceptions, or procedural steps.

10. At the end, provide a short "Sources" section based only
    on metadata returned by the search tool.

11. Never fabricate citations.

12. If the answer cannot be established from the retrieved
    policy content, recommend that the user confirm with the
    appropriate university office rather than guessing.
""",

        expected_output=(
            "A clear, accurate answer grounded in the university "
            "policy knowledge base, including relevant source "
            "metadata and an explicit statement when the "
            "available information is insufficient."
        ),

        agent=agent,
    )


# =========================================================
# Main function
# =========================================================

def ask_university_policy(question: str) -> str:
    """
    Send a university policy question to the CrewAI agent.
    """

    question = question.strip()

    if not question:
        return "Please enter a question."

    validate_configuration()

    agent = create_agent()

    task = create_task(
        agent=agent,
        question=question,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    return str(result)
