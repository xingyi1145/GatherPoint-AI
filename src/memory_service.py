from __future__ import annotations

from typing import Any


# -----------------------------------------------------------------------------
# Conversation State
# -----------------------------------------------------------------------------

def new_conversation() -> dict[str, Any]:
    """
    Create an empty conversation state for one GatherPoint planning session.

    app.py stores this dictionary in st.session_state. Recent messages remain
    directly available to both the chat UI and the local agent, while older
    turns can be compacted into the summary field.

    Returns:
        A conversation dictionary containing:
        - messages: Ordered recent chat messages with role and content fields.
        - summary: Optional compact representation of earlier conversation.
    """
    return {
        "messages": [],
        "summary": "",
    }


def _message_to_summary_line(message: dict[str, Any]) -> str:
    """
    Convert one chat message into a safe, compact summary line.

    This fallback is deterministic and does not require model inference. Person
    2 may later replace it with structured local-LLM summarization while
    preserving add_message()'s public behavior.

    Args:
        message: Message dictionary containing role and content fields.

    Returns:
        A concise text line suitable for conversation['summary'].
    """
    role = str(message.get("role", "user")).strip().title()
    content = " ".join(str(message.get("content", "")).split())

    return f"{role}: {content}"


def _append_to_summary(
    existing_summary: str,
    messages_to_compress: list[dict[str, Any]],
) -> str:
    """
    Merge older messages into the compressed conversation summary.

    The function deliberately performs no semantic interpretation. It preserves
    a readable chronological fallback when no local summarization model is
    available.

    Args:
        existing_summary: Previously compressed earlier context.
        messages_to_compress: Old messages removed from the raw message window.

    Returns:
        Updated compact summary text.
    """
    summary_lines = [
        _message_to_summary_line(message)
        for message in messages_to_compress
        if str(message.get("content", "")).strip()
    ]

    if not summary_lines:
        return existing_summary.strip()

    new_summary = "\n".join(summary_lines).strip()

    if not existing_summary.strip():
        return new_summary

    return f"{existing_summary.strip()}\n{new_summary}"


def add_message(
    conversation: dict[str, Any],
    role: str,
    content: str,
    max_recent_messages: int = 6,
) -> None:
    """
    Append a chat message and enforce a bounded recent-message window.

    When the message count exceeds max_recent_messages, the oldest messages are
    moved into conversation['summary']. This protects local vLLM or ReAct
    inference from unbounded prompt growth while maintaining basic continuity.

    Args:
        conversation: Mutable state created by new_conversation().
        role: Chat role. Only "user" and "assistant" are accepted.
        content: Message text to store.
        max_recent_messages: Maximum raw messages retained after compression.

    Returns:
        None. The conversation dictionary is modified in place.

    Raises:
        ValueError: If role is invalid or max_recent_messages is less than one.
    """
    if role not in {"user", "assistant"}:
        raise ValueError("role must be either 'user' or 'assistant'")

    if max_recent_messages < 1:
        raise ValueError("max_recent_messages must be at least 1")

    # Initialize required keys defensively for compatibility with restored
    # session state or externally created conversation dictionaries.
    conversation.setdefault("messages", [])
    conversation.setdefault("summary", "")

    # Store only stripped text to avoid retaining accidental whitespace-only turns.
    cleaned_content = str(content).strip()

    if not cleaned_content:
        return

    conversation["messages"].append(
        {
            "role": role,
            "content": cleaned_content,
        }
    )

    # Keep only the newest N raw messages. Older turns are retained in the
    # compressed summary, which is later included by agent_service.py.
    overflow_count = len(conversation["messages"]) - max_recent_messages

    if overflow_count <= 0:
        return

    messages_to_compress = conversation["messages"][:overflow_count]
    conversation["messages"] = conversation["messages"][overflow_count:]

    conversation["summary"] = _append_to_summary(
        existing_summary=str(conversation["summary"]),
        messages_to_compress=messages_to_compress,
    )


# -----------------------------------------------------------------------------
# Friend Profile Normalization
# -----------------------------------------------------------------------------

def _normalize_string_list(value: Any) -> list[str]:
    """
    Convert common storage formats into a clean list of non-empty strings.

    SQLite and JSON-backed Profile implementations may return strings, lists,
    tuples, sets, or None. The UI expects a list for dietary restrictions and
    interests, so this helper gives normalize_profile() one reliable shape.

    Args:
        value: Raw field value from persistent Profile storage.

    Returns:
        A list of stripped, non-empty string values.
    """
    if value is None:
        return []

    if isinstance(value, str):
        # Support both a single value and simple comma-separated fallback data.
        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    if isinstance(value, (list, tuple, set)):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    # Unknown scalar values are safely converted instead of breaking UI rendering.
    return [str(value).strip()] if str(value).strip() else []


def _normalize_availability(value: Any) -> dict[str, Any]:
    """
    Convert availability data into a dictionary safe for downstream rendering.

    Args:
        value: Raw availability record from profile storage.

    Returns:
        A dictionary. Invalid or absent availability data becomes an empty dict.
    """
    if isinstance(value, dict):
        return value

    return {}


def normalize_profile(raw_profile: dict[str, Any] | None) -> dict[str, Any]:
    """
    Normalize incomplete Friend Profile data into GatherPoint's UI contract.

    app.py calls this function before displaying member cards and active
    constraints. Every returned key has a safe default so a partially completed
    Profile record cannot crash the application.

    Expected normalized schema:
        {
            "name": str,
            "address": str,
            "latitude": float | None,
            "longitude": float | None,
            "transit_mode": str,
            "budget": str,
            "dietary_restrictions": list[str],
            "interests": list[str],
            "availability": dict,
        }

    Args:
        raw_profile: Profile record returned by SQLite, JSON, or another local
            storage implementation. None is accepted for defensive use.

    Returns:
        A complete normalized Profile dictionary.
    """
    raw_profile = raw_profile or {}

    return {
        "name": str(raw_profile.get("name") or "Unknown member").strip(),
        "address": str(raw_profile.get("address") or "Not specified").strip(),
        "latitude": raw_profile.get("latitude"),
        "longitude": raw_profile.get("longitude"),
        "transit_mode": str(
            raw_profile.get("transit_mode") or "unspecified"
        ).strip(),
        "budget": str(raw_profile.get("budget") or "unspecified").strip(),
        "dietary_restrictions": _normalize_string_list(
            raw_profile.get("dietary_restrictions")
        ),
        "interests": _normalize_string_list(raw_profile.get("interests")),
        "availability": _normalize_availability(
            raw_profile.get("availability")
        ),
    }


# -----------------------------------------------------------------------------
# Local Profile Storage Integration
# -----------------------------------------------------------------------------

def load_group_profiles(group_id: str) -> list[dict[str, Any]]:
    """
    Load normalized Friend Profiles for one GatherPoint group.

    This is the storage boundary that Person 2 should implement with local
    SQLite or JSON data. The function should query by group_id, normalize every
    record with normalize_profile(), and return an empty list for new groups.

    Suggested SQLite schema:
        CREATE TABLE group_members (
            group_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            name TEXT NOT NULL,
            address TEXT,
            latitude REAL,
            longitude REAL,
            transit_mode TEXT,
            budget TEXT,
            dietary_restrictions TEXT,
            interests TEXT,
            availability_json TEXT,
            PRIMARY KEY (group_id, member_id)
        );

    Args:
        group_id: Stable identifier of the active meetup workspace.

    Returns:
        A list of normalized Profile dictionaries. Returns an empty list when
        the group has no saved members.

    Raises:
        RuntimeError: Only for actual local-storage failures, such as a
            corrupted database or unreadable data file. Do not raise simply
            because the group has no stored Profiles.
    """
    # TODO(Person 2):
    # 1. Open a local SQLite database or JSON file.
    # 2. Query records that match group_id.
    # 3. Decode JSON fields such as dietary_restrictions and availability.
    # 4. Return [normalize_profile(record) for record in records].
    _ = group_id

    return []


# -----------------------------------------------------------------------------
# Long-Term Memory Extension Point
# -----------------------------------------------------------------------------

def save_group_memory(
    group_id: str,
    memory_text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save a durable group-planning fact for later retrieval.

    This optional extension point can persist facts such as venue rejections,
    standing dietary requirements, or repeated schedule preferences. Person 2
    may connect it to SQLite, JSON, or a local vector database.

    Args:
        group_id: Stable identifier of the relevant meetup group.
        memory_text: Concise factual planning memory to persist.
        metadata: Optional structured fields, such as source, timestamp, or
            memory category.

    Returns:
        None.

    Note:
        This placeholder intentionally performs no persistence until a local
        storage backend is selected.
    """
    # TODO(Person 2): Persist the memory locally and attach metadata needed for
    # later RAG ranking. Avoid storing secrets or unnecessary sensitive details.
    _ = group_id
    _ = memory_text
    _ = metadata


def search_group_memory(
    group_id: str,
    query: str,
    limit: int = 3,
) -> list[str]:
    """
    Retrieve concise long-term group memories relevant to a user request.

    agent_service.retrieve_relevant_memories() can call this function after
    Person 2 implements a local keyword, embedding, or hybrid search strategy.

    Args:
        group_id: Stable identifier of the active meetup group.
        query: Latest user request or extracted planning constraints.
        limit: Maximum number of memory snippets to return.

    Returns:
        A ranked list of short memory strings. Returns an empty list when no
        relevant local memory exists.
    """
    # TODO(Person 2):
    # 1. Search only records belonging to group_id.
    # 2. Rank by relevance to query.
    # 3. Return no more than limit short, factual snippets.
    _ = group_id
    _ = query
    _ = limit

    return []