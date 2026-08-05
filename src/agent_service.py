from __future__ import annotations

import logging
import os
import re
from typing import Any

import requests

from react_meetup_agent_poc import build_agent
from memory_service import load_group_profiles, normalize_profile


RAG_SERVER_URL = os.getenv(
    "GATHERPOINT_RAG_SERVER_URL",
    "http://127.0.0.1:8000",
)
RAG_RETRIEVE_PROFILES_URL = f"{RAG_SERVER_URL.rstrip('/')}/retrieve_profiles"


# 1. Quick and easy logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent_debug.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _extract_forced_final_answer(text: str) -> str | None:
    """
    Recover a valid final answer when the model hallucinates an action name.

    Some local models emit:
    - Action: Final Answer
    - Action Input: <summary>

    LangChain treats that as an invalid tool and keeps looping. This helper
    detects the pattern and extracts the intended user-facing summary.
    """
    if not text:
        return None

    direct_final = re.search(
        r"(?is)final\s*answer\s*:\s*(.+)$",
        text,
    )
    if direct_final:
        candidate = direct_final.group(1).strip()
        if candidate:
            return candidate

    action_match = re.search(
        r"(?is)action\s*:\s*(final\s*answer|final_answer)\b",
        text,
    )
    if not action_match:
        return None

    action_input_match = re.search(
        r"(?is)action\s*input\s*:\s*(.+?)(?:\n\s*(?:observation|thought|action|final\s*answer)\s*:|$)",
        text,
    )
    if action_input_match:
        candidate = action_input_match.group(1).strip().strip('"').strip("'")
        if candidate:
            return candidate

    tail_after_action = text[action_match.end():].strip()
    return tail_after_action or None


# -----------------------------------------------------------------------------
# Context Formatting Utilities
# -----------------------------------------------------------------------------

def _format_profile_context(profiles: list[dict[str, Any]]) -> str:
    """
    Convert normalized Friend Profiles into trusted planning context.

    This text is intended for the local meetup-planning agent. It makes each
    member's location, mobility preference, budget, dietary requirements, and
    availability explicit so the agent can avoid inventing user constraints.

    Args:
        profiles: Normalized Profile records loaded for the active group.

    Returns:
        A readable multi-line string suitable for a system or planning prompt.
    """
    if not profiles:
        # Return nothing! Do not confuse the LLM with missing data instructions.
        return ""

    lines = [
        "Trusted Friend Profiles:",
        "=" * 60,
    ]

    for raw_profile in profiles:
        profile = normalize_profile(raw_profile)

        dietary = ", ".join(profile["dietary_restrictions"]) or "none"
        interests = ", ".join(profile["interests"]) or "not specified"
        availability = (
            ", ".join(
                f"{day}: {status}"
                for day, status in profile["availability"].items()
            )
            or "not specified"
        )

        lines.append(
            f"- {profile['name']}: "
            f"location={profile['address']}; "
            f"transit={profile['transit_mode']}; "
            f"budget={profile['budget']}; "
            f"dietary={dietary}; "
            f"interests={interests}; "
            f"availability={availability}"
        )

    return "\n".join(lines)


def _format_memory_context(memories: list[str]) -> str:
    """
    Convert retrieved long-term memory snippets into a prompt-ready block.

    Memories may include prior venue rejections, group preferences, scheduling
    habits, or constraints learned in earlier sessions. The agent should treat
    these as useful context, but should prioritize the user's latest request.

    Args:
        memories: Relevant memory snippets returned by a RAG/storage layer.

    Returns:
        A formatted string for the local planning prompt.
    """
    if not memories:
        # Return nothing! 
        return ""

    lines = [
        "Relevant Group Memory:",
        "=" * 60,
    ]

    for index, memory in enumerate(memories, start=1):
        lines.append(f"{index}. {memory}")

    return "\n".join(lines)


def _format_conversation_context(conversation: dict[str, Any]) -> str:
    """
    Build a bounded context block from session conversation state.

    memory_service.py retains a small recent-message window and can compress
    older turns into conversation['summary']. This prevents local inference
    context from growing indefinitely while preserving key earlier decisions.

    Args:
        conversation: Conversation dictionary maintained by app.py.

    Returns:
        A formatted string containing optional summary and recent messages.
    """
    summary = str(conversation.get("summary", "")).strip()
    messages = conversation.get("messages", [])

    lines = [
        "Conversation Context:",
        "=" * 60,
    ]

    if summary:
        lines.append("Compressed earlier context:")
        lines.append(summary)
        lines.append("")

    if messages:
        lines.append("Recent turns:")

        for message in messages:
            role = str(message.get("role", "user")).title()
            content = str(message.get("content", "")).strip()

            if content:
                lines.append(f"{role}: {content}")
    else:
        lines.append("No earlier messages are available.")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Persistent Memory Retrieval
# -----------------------------------------------------------------------------

def retrieve_relevant_memories(
    group_id: str,
    user_message: str,
) -> list[str]:
    """
    Retrieve long-term memory relevant to the current planning request.

    This calls the remote GPU-backed RAG microservice. If the service is
    unreachable or returns an invalid payload, the function falls back to an
    empty list so the UI can still render safely.

    Args:
        group_id: Stable identifier of the active meetup group.
        user_message: The latest user request used as the retrieval query.

    Returns:
        A list of concise memory strings. Return an empty list when no relevant
        information exists; do not fail merely because a new group has no memory.
    """
    try:
        query_text = f"group_id={group_id}\n{user_message}"
        response = requests.post(
            RAG_RETRIEVE_PROFILES_URL,
            json={"query": query_text},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        logger.exception(
            "Remote RAG service request failed: %s",
            RAG_RETRIEVE_PROFILES_URL,
        )
        return []
    except ValueError:
        logger.exception("Remote RAG service returned invalid JSON.")
        return []

    matches = payload.get("matches", [])

    if not isinstance(matches, list) or not matches:
        return []

    contexts: list[str] = []
    for match in matches:
        if not isinstance(match, dict):
            continue

        document = str(match.get("document", "")).strip()
        if document:
            contexts.append(document)

    return contexts


# -----------------------------------------------------------------------------
# Local Agent Integration
# -----------------------------------------------------------------------------
def _build_planning_prompt(
    group_id: str,
    user_message: str,
    conversation: dict[str, Any],
    profiles: list[dict[str, Any]],
    memories: list[str],
) -> str:
    """
    Construct the bounded local-LLM prompt for one planning turn.

    The prompt combines trusted Profile data, retrieved memory, compact
    conversation history, and the latest request. To reduce instruction
    override in smaller local models, optional context sections are injected
    only when real data exists, and the latest user request is placed last.

    Args:
        group_id: Active group identifier.
        user_message: Latest request from the user.
        conversation: Current compact conversation state.
        profiles: Normalized Profiles loaded for the group.
        memories: Relevant retrieved-memory snippets.

    Returns:
        A prompt string ready to send to a local planning agent or LLM.
    """
    profile_context = _format_profile_context(profiles)
    memory_context = _format_memory_context(memories)
    conversation_context = _format_conversation_context(conversation)

    sections: list[str] = [
        "You are GatherPoint, a local-first group meetup planning assistant.",
        "",
        "Your job is to recommend practical meetup options that work fairly for the",
        "whole group. Use trusted Friend Profiles, retrieved group memory, recent",
        "conversation, and GIS tool results when available.",
        "",
        "Rules:",
        "- Do not invent locations, budgets, dietary needs, schedules, or travel modes.",
        "- Treat explicit user instructions in the latest request as the highest priority.",
        "- Respect dietary restrictions and identify conflicts or missing information.",
        "- Prefer fair travel-time tradeoffs instead of optimizing only for one person.",
        "- If no practical common meeting area exists, explain why and suggest a concrete",
        "  relaxation, such as a longer travel limit, a different venue type, or a date.",
        "- Keep the final answer concise and user-facing.",
        "- When recommendations are available, explain the top choices and why they fit.",
        "- CRITICAL STOPPING RULE: Once you receive the list of places, DO NOT write 'Action: Final Answer'. You must immediately output the exact text 'Final Answer: ' followed directly by your friendly summary.",
        "",
        "Active group ID:",
        group_id,
    ]

    if profile_context:
        sections.extend(["", profile_context])

    if memory_context:
        sections.extend(["", memory_context])

    sections.extend([
        "",
        conversation_context,
        "",
        "Latest user request:",
        user_message,
    ])

    return "\n".join(sections).strip()


def _invoke_local_planning_agent(
    prompt: str,
) -> str:
    try:
        logger.debug("Invoking local planning agent.")
        executor = build_agent()
        # Request intermediate steps so we can recover malformed stop outputs.
        executor.return_intermediate_steps = True
        result = executor.invoke({"input": prompt})
        logger.debug("Raw agent invocation payload: %s", result)
    except Exception as e:
        # Catch errors gracefully so the UI doesn't completely break
        logger.exception("Local agent invocation failed.")
        recovered = _extract_forced_final_answer(str(e))
        if recovered:
            logger.warning(
                "Recovered final answer from malformed action in exception path."
            )
            return recovered
        raise RuntimeError(f"Local agent error: {str(e)}")

    if isinstance(result, dict):
        output = str(result.get("output", "")).strip()

        intermediate_steps = result.get("intermediate_steps", [])
        for step in reversed(intermediate_steps):
            if not isinstance(step, tuple) or len(step) < 1:
                continue

            action = step[0]
            tool_name = str(getattr(action, "tool", "")).strip()

            if tool_name in {"Final Answer", "final_answer"}:
                tool_input = str(getattr(action, "tool_input", "")).strip()
                if tool_input:
                    logger.warning(
                        "Recovered final answer from hallucinated tool action '%s'.",
                        tool_name,
                    )
                    return tool_input

        recovered_from_output = _extract_forced_final_answer(output)
        if recovered_from_output:
            logger.warning("Recovered final answer from raw model output text.")
            return recovered_from_output
    else:
        output = str(result).strip()
        recovered_from_output = _extract_forced_final_answer(output)
        if recovered_from_output:
            logger.warning("Recovered final answer from raw non-dict output text.")
            return recovered_from_output

    if not output:
        raise RuntimeError("Local agent returned an empty response.")

    return output


# -----------------------------------------------------------------------------
# Main Agent Turn
# -----------------------------------------------------------------------------

def run_agent_turn(
    group_id: str,
    user_message: str,
    conversation: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """
    Execute one context-aware GatherPoint meetup-planning turn.
    """
    if not group_id.strip():
        raise RuntimeError("A group workspace ID is required before planning.")

    if not user_message.strip():
        raise RuntimeError("Please enter a meetup-planning request.")

    logger.info("--- NEW TURN STARTED ---")
    logger.info("Group ID: %s", group_id)
    logger.info("User Message: %s", user_message)

    # 1. Load persistent Profile data
    raw_profiles = load_group_profiles(group_id)
    profiles = [normalize_profile(profile) for profile in raw_profiles]
    logger.debug("Loaded profile count: %d", len(profiles))

    # 2. Retrieve only memory relevant to this request
    memories = retrieve_relevant_memories(group_id, user_message)
    logger.debug("Retrieved memory count: %d", len(memories))

    # 3. Build a single bounded context block for the downstream local agent.
    planning_prompt = _build_planning_prompt(
        group_id=group_id,
        user_message=user_message,
        conversation=conversation,
        profiles=profiles,
        memories=memories,
    )
    logger.debug("LLM Prompt:\n%s", planning_prompt)

    # 4. Call the local model/agent.
    answer = _invoke_local_planning_agent(planning_prompt)
    logger.info("LLM Raw Response: %s", answer)

    if not answer or not answer.strip():
        logger.warning("Agent returned an empty response.")
        raise RuntimeError(
            "The local planning agent returned an empty response. "
            "Please check the model endpoint and agent integration."
        )

    logger.info("Agent successfully reached stopping condition.")

    # 5. app.py requires this exact tuple shape.
    return answer.strip(), profiles, memories