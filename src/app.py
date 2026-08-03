from __future__ import annotations

import streamlit as st

from agent_service import run_agent_turn
from memory_service import add_message, new_conversation, normalize_profile


# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
# Must be called before any other Streamlit rendering command.
st.set_page_config(
    page_title="GatherPoint | Plan together",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Product UI Theme
# -----------------------------------------------------------------------------
# Use custom CSS instead of Streamlit's default visual style so the demo looks
# like a cohesive local-first meetup-planning product rather than a chat demo.
st.markdown(
    """
    <style>
        :root {
            --bg: #07111f;
            --panel: #0d1b2d;
            --panel-soft: #10243a;
            --line: rgba(158, 185, 216, 0.16);
            --text: #eef6ff;
            --muted: #93a8c0;
            --green: #39d7aa;
            --purple: #8891ff;
            --amber: #ffbd66;
            --red: #ff7e8b;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 12% -10%,
                    rgba(55, 215, 170, 0.13),
                    transparent 32rem
                ),
                radial-gradient(
                    circle at 92% 10%,
                    rgba(136, 145, 255, 0.16),
                    transparent 30rem
                ),
                var(--bg);
            color: var(--text);
        }

        /* Hide Streamlit's default application chrome for a cleaner demo UI. */
        #MainMenu, footer, header {
            visibility: hidden;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* Restyle the sidebar as a persistent workspace panel. */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #091727 0%, #07111f 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.3rem;
        }

        h1, h2, h3, p, span, label {
            color: var(--text);
        }

        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(
                135deg,
                rgba(18, 43, 68, 0.96),
                rgba(12, 28, 47, 0.92)
            );
            border-radius: 22px;
            padding: 1.45rem 1.6rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.19);
        }

        .eyebrow {
            color: var(--green);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .hero-title {
            color: var(--text);
            font-size: 2rem;
            font-weight: 750;
            line-height: 1.1;
            margin: 0;
        }

        .hero-subtitle {
            color: var(--muted);
            margin-top: 0.55rem;
            margin-bottom: 0;
            line-height: 1.55;
        }

        .local-badge {
            display: inline-block;
            border: 1px solid rgba(57, 215, 170, 0.45);
            background: rgba(57, 215, 170, 0.11);
            color: #8df2d1;
            border-radius: 99px;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.35rem 0.7rem;
            margin-top: 0.8rem;
        }

        .section-label {
            color: #b8c9dd;
            font-size: 0.74rem;
            font-weight: 750;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            margin: 0.3rem 0 0.6rem 0;
        }

        .panel {
            border: 1px solid var(--line);
            background: rgba(13, 27, 45, 0.88);
            border-radius: 18px;
            padding: 1rem;
            margin-bottom: 0.9rem;
        }

        .profile-card {
            border-left: 3px solid var(--green);
            background: rgba(20, 46, 71, 0.64);
            border-radius: 11px;
            padding: 0.72rem 0.78rem;
            margin-bottom: 0.62rem;
        }

        .profile-name {
            color: var(--text);
            font-size: 0.93rem;
            font-weight: 700;
        }

        .profile-detail {
            color: var(--muted);
            font-size: 0.77rem;
            line-height: 1.55;
            margin-top: 0.2rem;
        }

        .metric-card {
            border: 1px solid var(--line);
            background: rgba(16, 36, 58, 0.72);
            border-radius: 14px;
            padding: 0.7rem;
            text-align: center;
        }

        .metric-value {
            color: var(--green);
            font-size: 1.3rem;
            font-weight: 800;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.7rem;
            margin-top: 0.18rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .plan-item {
            border: 1px solid var(--line);
            background: linear-gradient(
                135deg,
                rgba(22, 48, 74, 0.88),
                rgba(13, 30, 49, 0.88)
            );
            border-radius: 13px;
            padding: 0.8rem;
            margin-top: 0.65rem;
        }

        .plan-number {
            display: inline-block;
            color: #07111f;
            background: var(--green);
            width: 1.5rem;
            height: 1.5rem;
            text-align: center;
            border-radius: 50%;
            font-size: 0.78rem;
            line-height: 1.5rem;
            font-weight: 800;
            margin-right: 0.45rem;
        }

        .plan-title {
            color: var(--text);
            font-weight: 700;
            font-size: 0.88rem;
        }

        .tag {
            display: inline-block;
            border-radius: 99px;
            background: rgba(136, 145, 255, 0.15);
            border: 1px solid rgba(136, 145, 255, 0.35);
            color: #c4c8ff;
            padding: 0.2rem 0.48rem;
            margin: 0.36rem 0.24rem 0 0;
            font-size: 0.68rem;
            font-weight: 650;
        }

        .memory-item {
            border-left: 2px solid var(--purple);
            background: rgba(136, 145, 255, 0.08);
            color: #c8d2e1;
            font-size: 0.78rem;
            padding: 0.55rem 0.65rem;
            margin-bottom: 0.45rem;
            border-radius: 0 8px 8px 0;
            line-height: 1.45;
        }

        /* Style user and assistant messages as product chat cards. */
        [data-testid="stChatMessage"] {
            border: 1px solid var(--line);
            background: rgba(12, 28, 47, 0.72);
            border-radius: 14px;
            margin-bottom: 0.7rem;
            padding: 0.2rem 0.3rem;
        }

        [data-testid="stChatInput"] {
            border: 1px solid rgba(57, 215, 170, 0.36);
            border-radius: 14px;
            background: rgba(12, 29, 47, 0.9);
        }

        .welcome-card {
            border: 1px dashed rgba(57, 215, 170, 0.42);
            background: rgba(57, 215, 170, 0.06);
            border-radius: 16px;
            padding: 1.1rem;
            color: #d8e8f6;
            margin: 0.35rem 0 1rem 0;
        }

        .welcome-card strong {
            color: #8df2d1;
        }

        div.stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(57, 215, 170, 0.45);
            background: rgba(57, 215, 170, 0.10);
            color: #dffef4;
            font-weight: 650;
        }

        div.stButton > button:hover {
            background: rgba(57, 215, 170, 0.20);
            border-color: var(--green);
            color: white;
        }

        [data-testid="stTextInput"] input {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session-State Initialization
# -----------------------------------------------------------------------------

def initialize_state() -> None:
    """
    Initialize all Streamlit session-state keys required by the workspace.

    Streamlit reruns this file after each interaction. Values stored in
    st.session_state persist across reruns, which allows the chat interface,
    active group, retrieved memories, and live-plan panel to stay synchronized.
    """
    if "group_id" not in st.session_state:
        # Default workspace ID used until the user selects another group.
        st.session_state.group_id = "friday-dinner-crew"

    if "conversation" not in st.session_state:
        # Conversation state is created by memory_service.py.
        st.session_state.conversation = new_conversation()

    if "loaded_profiles" not in st.session_state:
        # Contains normalized Profile records returned by the latest agent turn.
        st.session_state.loaded_profiles = []

    if "retrieved_memories" not in st.session_state:
        # Contains relevant long-term memory/RAG snippets for the latest turn.
        st.session_state.retrieved_memories = []

    if "last_answer" not in st.session_state:
        # Used by the Live Plan panel to indicate whether a response exists.
        st.session_state.last_answer = ""


def reset_conversation() -> None:
    """
    Start a new planning conversation without changing the selected group.

    Friend Profile data remains associated with the selected group. Only
    short-term chat history, retrieved-memory display, and response status
    are cleared.
    """
    st.session_state.conversation = new_conversation()
    st.session_state.retrieved_memories = []
    st.session_state.last_answer = ""


# -----------------------------------------------------------------------------
# Reusable UI Components
# -----------------------------------------------------------------------------

def profile_card(raw_profile: dict) -> None:
    """
    Render a compact Friend Profile card in the left workspace sidebar.

    The profile is normalized before rendering so missing values from SQLite,
    JSON, or future profile storage do not break the interface.

    Args:
        raw_profile: A potentially incomplete Friend Profile dictionary.
    """
    profile = normalize_profile(raw_profile)

    dietary = ", ".join(profile["dietary_restrictions"]) or "No restrictions"
    interests = ", ".join(profile["interests"]) or "Not specified"

    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-name">● {profile["name"]}</div>
            <div class="profile-detail">
                📍 {profile["address"]}<br>
                🚇 {profile["transit_mode"]} &nbsp; · &nbsp; 💳 {profile["budget"]}<br>
                🥗 {dietary}<br>
                ✨ {interests}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_plan() -> None:
    """
    Render the right-side Live Plan panel.

    The panel makes GatherPoint's reasoning inputs visible during a demo:
    group size, number of conversation turns, profile-derived constraints,
    relevant retrieved memory, compressed older context, and recommendation
    readiness.

    The final venue ranking can later be added here by extending
    agent_service.run_agent_turn() to return structured venue records.
    """
    profiles = st.session_state.loaded_profiles
    messages = st.session_state.conversation.get("messages", [])
    memories = st.session_state.retrieved_memories
    summary = st.session_state.conversation.get("summary", "")

    st.markdown('<p class="section-label">Live Plan</p>', unsafe_allow_html=True)

    # Display three compact metrics at the top of the planning panel.
    metric_one, metric_two, metric_three = st.columns(3)

    with metric_one:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(profiles)}</div>
                <div class="metric-label">Friends</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_two:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(messages) // 2}</div>
                <div class="metric-label">Turns</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_three:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">Local</div>
                <div class="metric-label">Inference</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Explain the planning context that should be used by the orchestration layer.
    st.markdown(
        """
        <div class="panel">
            <div class="section-label">Decision context</div>
            <div style="color:#cbd9e7; font-size:0.84rem; line-height:1.6;">
                GatherPoint combines friend preferences, recent conversation,
                retrieved memory, and GIS results before recommending a plan.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Convert active Profile fields into visible decision-constraint tags.
    if profiles:
        dietary_tags = []
        transit_tags = []

        for raw_profile in profiles:
            profile = normalize_profile(raw_profile)

            dietary_tags.extend(profile["dietary_restrictions"])

            if profile["transit_mode"] != "unspecified":
                transit_tags.append(profile["transit_mode"])

        st.markdown(
            '<p class="section-label">Active constraints</p>',
            unsafe_allow_html=True,
        )

        all_tags = sorted(set(dietary_tags + transit_tags))

        if all_tags:
            st.markdown(
                "".join(f'<span class="tag">{tag}</span>' for tag in all_tags),
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "Profile constraints will appear here after Person 2 connects storage."
            )
    else:
        st.info("Connect Friend Profiles to populate the group context.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p class="section-label">Memory retrieval</p>',
        unsafe_allow_html=True,
    )

    # Display RAG/long-term-memory results returned by agent_service.py.
    if memories:
        for memory in memories:
            st.markdown(
                f'<div class="memory-item">{memory}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No long-term memory retrieved for the current request.")

    # Older conversation can be compacted by memory_service.py to keep the
    # local LLM prompt within its available inference context window.
    if summary:
        with st.expander("View compressed earlier conversation"):
            st.write(summary)

    # This status card appears after the first successful assistant response.
    if st.session_state.last_answer:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<p class="section-label">Recommendation status</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="plan-item">
                <span class="plan-number">✓</span>
                <span class="plan-title">Agent response generated</span>
                <div class="profile-detail">
                    The latest recommendation is shown in the conversation panel.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Application Initialization
# -----------------------------------------------------------------------------
initialize_state()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
# The header communicates the product's core value proposition and local
# AMD/ROCm inference story before a judge or user starts interacting.
header_left, header_right = st.columns([5, 1.35])

with header_left:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Local-first group coordination</div>
            <h1 class="hero-title">GatherPoint</h1>
            <p class="hero-subtitle">
                Turn your group’s locations, schedules, budgets, transit preferences,
                and food constraints into a meetup plan that works for everyone.
            </p>
            <span class="local-badge">
                ● AMD ROCm · Local vLLM · Context-aware planning
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown("<br>", unsafe_allow_html=True)

    # Reset only the planning session, preserving the selected group ID.
    if st.button("↻ New plan", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.caption(
        "Model output is constrained to 256 tokens. Conversation history is "
        "compacted for local 4,096-token inference."
    )


# -----------------------------------------------------------------------------
# Group Workspace Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📍 Workspace")
    st.caption("Manage the meetup group that GatherPoint should remember.")

    # The group ID becomes the key used for Profile storage and memory lookup.
    group_id = st.text_input(
        "Group workspace",
        value=st.session_state.group_id,
        placeholder="friday-dinner-crew",
    ).strip()

    # If the user changes workspace, clear old visible Profiles to avoid
    # mistakenly showing profiles from the previously selected group.
    if group_id and group_id != st.session_state.group_id:
        st.session_state.group_id = group_id
        st.session_state.loaded_profiles = []

    st.markdown("---")
    st.markdown("### Group members")

    # Real Profile records appear after agent_service.py connects persistence.
    if st.session_state.loaded_profiles:
        for raw_profile in st.session_state.loaded_profiles:
            profile_card(raw_profile)
    else:
        st.info(
            "Profiles will appear here when Person 2 connects the SQLite/JSON "
            "profile service."
        )

        # These preview cards demonstrate the intended UI without pretending
        # that mock users are persistent data.
        st.markdown(
            """
            <div class="profile-card">
                <div class="profile-name">Preview: Alice</div>
                <div class="profile-detail">
                    📍 Union Station, Toronto<br>
                    🚇 WALK · 💳 $$<br>
                    🥗 Vegetarian
                </div>
            </div>
            <div class="profile-card">
                <div class="profile-name">Preview: Bob</div>
                <div class="profile-detail">
                    📍 Yonge & Bloor, Toronto<br>
                    🚇 TRANSIT · 💳 $<br>
                    🥗 No restrictions
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Local memory")

    # Recent messages and summary availability make multi-turn memory observable.
    recent_messages = len(st.session_state.conversation.get("messages", []))
    summary_exists = bool(st.session_state.conversation.get("summary", ""))

    st.caption(f"Recent messages retained: {recent_messages}/6")
    st.caption(
        f"Earlier summary: {'available' if summary_exists else 'not needed yet'}"
    )
    st.caption("RAG retrieval: ready for Person 1 integration")


# -----------------------------------------------------------------------------
# Main Three-Panel Workspace
# -----------------------------------------------------------------------------
# The sidebar is the first panel. The main area contains conversation and
# Live Plan columns, creating the product's core planning workspace.
chat_column, plan_column = st.columns([1.75, 1], gap="large")


# -----------------------------------------------------------------------------
# Conversation Panel
# -----------------------------------------------------------------------------
with chat_column:
    st.markdown('<p class="section-label">Conversation</p>', unsafe_allow_html=True)

    # Empty-state guidance helps the user understand the product before the
    # first turn and supplies demo-friendly starter prompts.
    if not st.session_state.conversation["messages"]:
        st.markdown(
            """
            <div class="welcome-card">
                <strong>Ready to coordinate the group.</strong><br><br>
                Ask naturally. GatherPoint should remember the selected group’s
                profiles and the current conversation across turns.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="section-label">Try a starting point</p>',
            unsafe_allow_html=True,
        )

        chip_one, chip_two = st.columns(2)

        with chip_one:
            if st.button(
                "🥗 Find a vegetarian dinner for Friday",
                use_container_width=True,
            ):
                # Store the button request and process it in the normal chat flow.
                st.session_state.prefill_prompt = (
                    "Find a vegetarian dinner option for our group on Friday evening."
                )

        with chip_two:
            if st.button(
                "⏱ Keep everyone under 30 minutes away",
                use_container_width=True,
            ):
                st.session_state.prefill_prompt = (
                    "Find a meetup option that keeps everyone's travel time "
                    "under 30 minutes."
                )

    # Re-render prior turns stored in the conversation state.
    for message in st.session_state.conversation["messages"]:
        avatar = "🧑" if message["role"] == "user" else "📍"

        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # The chat input is the standard free-form entry point.
    prompt = st.chat_input(
        "Ask GatherPoint to plan a meetup...",
        key="chat_input",
    )

    # Prompt chips use the same agent path as direct user chat input.
    if not prompt and st.session_state.get("prefill_prompt"):
        prompt = st.session_state.pop("prefill_prompt")

    if prompt:
        # Save the user request before calling the agent so the current request
        # is available in the short-term context sent to local inference.
        add_message(st.session_state.conversation, "user", prompt)

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="📍"):
            # The status component transparently demonstrates each product step
            # during live judging without exposing chain-of-thought.
            status = st.status("GatherPoint is planning...", expanded=True)

            try:
                status.write("Loading Friend Profiles and saved group context...")
                status.write(
                    "Preparing a bounded multi-turn prompt for local inference..."
                )
                status.write(
                    "Checking available meetup tools and constraints..."
                )

                # Service-layer contract:
                # answer: UI-ready assistant text
                # profiles: normalized group Profiles
                # memories: retrieved RAG/long-term-memory snippets
                answer, profiles, memories = run_agent_turn(
                    group_id=st.session_state.group_id,
                    user_message=prompt,
                    conversation=st.session_state.conversation,
                )

                # Update all Live Plan data from the successful agent turn.
                st.session_state.loaded_profiles = profiles
                st.session_state.retrieved_memories = memories
                st.session_state.last_answer = answer

                status.update(
                    label="Recommendation ready",
                    state="complete",
                    expanded=False,
                )
                st.markdown(answer)

            except RuntimeError as exc:
                # RuntimeError is the expected service-layer failure type for
                # unavailable local inference, storage, RAG, or GIS services.
                answer = (
                    "⚠️ **GatherPoint could not complete this request.**\n\n"
                    f"{exc}"
                )

                status.update(
                    label="Unable to complete request",
                    state="error",
                    expanded=False,
                )
                st.error(answer)

        # Store the assistant response even if an expected runtime failure
        # occurs, so the conversation accurately records the interaction.
        add_message(st.session_state.conversation, "assistant", answer)


# -----------------------------------------------------------------------------
# Live Plan Panel
# -----------------------------------------------------------------------------
with plan_column:
    render_live_plan()