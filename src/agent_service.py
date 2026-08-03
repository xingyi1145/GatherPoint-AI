from __future__ import annotations

from typing import Any

from memory_service import load_group_profiles, normalize_profile


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
        return (
            "No persistent Friend Profiles are available for this group. "
            "Ask the user for the participants' locations, travel modes, "
            "budget, dietary needs, and preferred time."
        )

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
        return "No relevant long-term memory was retrieved for this request."

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

    Person 2 should replace this temporary implementation with a local RAG,
    SQLite, JSON, or vector-store lookup. It should return short, factual
    snippets that help with planning, such as a prior venue rejection or a
    recurring dietary constraint.

    Args:
        group_id: Stable identifier of the active meetup group.
        user_message: The latest user request used as the retrieval query.

    Returns:
        A list of concise memory strings. Return an empty list when no relevant
        information exists; do not fail merely because a new group has no memory.
    """
    # TODO(Person 2):
    # 1. Query local persistent memory by group_id.
    # 2. Rank records against user_message.
    # 3. Return only short, relevant, non-sensitive snippets.
    _ = group_id
    _ = user_message

    return []


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
    conversation history, and the latest request. It instructs the downstream
    agent not to fabricate user information and to explain constraint tradeoffs.

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

    return f"""
You are GatherPoint, a local-first group meetup planning assistant.

Your job is to recommend practical meetup options that work fairly for the
whole group. Use trusted Friend Profiles, retrieved group memory, recent
conversation, and GIS tool results when available.

Rules:
- Do not invent locations, budgets, dietary needs, schedules, or travel modes.
- Treat explicit user instructions in the latest request as the highest priority.
- Respect dietary restrictions and identify conflicts or missing information.
- Prefer fair travel-time tradeoffs instead of optimizing only for one person.
- If no practical common meeting area exists, explain why and suggest a concrete
  relaxation, such as a longer travel limit, a different venue type, or a date.
- Keep the final answer concise and user-facing.
- When recommendations are available, explain the top choices and why they fit.

Active group ID:
{group_id}

{profile_context}

{memory_context}

{conversation_context}

Latest user request:
{user_message}
""".strip()


def _invoke_local_planning_agent(
    prompt: str,
) -> str:
    """
    Invoke the local GatherPoint LLM/agent implementation.

    This function is deliberately isolated so Person 2 can connect the existing
    ReAct agent, a direct vLLM/OpenAI-compatible endpoint, or a local LangChain
    workflow without changing app.py or the public run_agent_turn() contract.

    Args:
        prompt: Fully assembled bounded planning prompt.

    Returns:
        Assistant text ready for the Streamlit conversation panel.

    Raises:
        RuntimeError: When local inference has not been connected or fails.
    """
    # TODO(Person 2):
    # Recommended implementation options:
    # - Import build_agent() from react_meetup_agent_poc.py and invoke it.
    # - Call a local AMD ROCm/vLLM OpenAI-compatible endpoint.
    # - Pass the resulting recommendation through GIS tools where appropriate.
    #
    # Keep failures as RuntimeError so app.py can show a safe user-facing error.
    _ = prompt

    raise RuntimeError(
        "Local planning inference is not connected yet. "
        "Connect this function to the GatherPoint ReAct agent or local vLLM "
        "endpoint, then return the agent's final answer."
    )


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

    Workflow:
    1. Load persistent Friend Profiles for the selected group.
    2. Normalize every Profile for safe UI and prompt usage.
    3. Retrieve relevant long-term group memory.
    4. Build a bounded planning prompt from profiles, memory, and chat context.
    5. Invoke the local planning agent.
    6. Return data required by both the chat panel and Live Plan panel.

    Args:
        group_id: Stable ID of the active meetup workspace, for example
            "friday-dinner-crew".
        user_message: Latest natural-language planning request.
        conversation: Conversation state maintained by memory_service.py.

    Returns:
        A three-item tuple:
        - answer: Assistant response ready for Streamlit markdown rendering.
        - profiles: Normalized Friend Profiles used during this planning turn.
        - memories: Retrieved long-term-memory snippets used during this turn.

    Raises:
        RuntimeError: If profile loading, memory retrieval, GIS, or local
            inference cannot complete the request.
    """
    if not group_id.strip():
        raise RuntimeError("A group workspace ID is required before planning.")

    if not user_message.strip():
        raise RuntimeError("Please enter a meetup-planning request.")

    # Load persistent Profile data first because it is the trusted basis for
    # fair location, transit, budget, and dietary decision-making.
    raw_profiles = load_group_profiles(group_id)
    profiles = [normalize_profile(profile) for profile in raw_profiles]

    # Retrieve only memory relevant to this request; unrelated historical data
    # should not unnecessarily increase local inference context.
    memories = retrieve_relevant_memories(group_id, user_message)

    # Build a single bounded context block for the downstream local agent.
    planning_prompt = _build_planning_prompt(
        group_id=group_id,
        user_message=user_message,
        conversation=conversation,
        profiles=profiles,
        memories=memories,
    )

    # Call the local model/agent. The implementation may use the project's
    # existing ReAct + GIS workflow after Person 2 completes integration.
    answer = _invoke_local_planning_agent(planning_prompt)

    if not answer or not answer.strip():
        raise RuntimeError(
            "The local planning agent returned an empty response. "
            "Please check the model endpoint and agent integration."
        )

    # app.py requires this exact tuple shape.
    return answer.strip(), profiles, memories