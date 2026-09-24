import json
import os
from datetime import datetime

from crewai.tools import tool


PENDING_FILE = "pending_escalations.json"


@tool("Human Escalation")
def escalation_tool(
    user_issue: str,
) -> str:
    """
    Escalate a university policy issue to human support.
    """

    escalation = {
        "timestamp": datetime.utcnow().isoformat(),
        "issue": user_issue,
        "status": "Pending",
    }

    existing = []

    if os.path.exists(PENDING_FILE):

        try:
            with open(
                PENDING_FILE,
                "r",
                encoding="utf-8",
            ) as file:
                existing = json.load(file)

        except Exception:
            existing = []

    existing.append(escalation)

    with open(
        PENDING_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            existing,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return (
        "The request has been escalated to human "
        "support and is now marked as Pending."
    )
