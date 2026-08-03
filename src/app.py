from __future__ import annotations

import streamlit as st

from agent_service import run_agent_turn
from memory_service import (
    add_message,
    load_group_profiles,
    new_conversation,
    normalize_profile,
)


# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
# Must be called before any other Streamlit rendering command.
st.set_page_config(
    page_title="GatherPoint | Plan together",
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
        /* --- FORCE SIDEBAR TO STAY OPEN --- */
        /* Hide the inside collapse button (<<) */
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }

        /* 1. Completely hide the collapse/expand arrow */
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        /* 2. Force the sidebar to remain visible and locked in place */
        [data-testid="stSidebar"] {
            display: flex !important;
            transform: none !important;
            visibility: visible !important;
            min-width: 16rem !important;
        }

        /* 3. Ensure the main chat area doesn't overlap the locked sidebar */
        [data-testid="stSidebar"] + section {
            margin-left: 0 !important;
        }

        /* 4. Force the sidebar to remain visible and locked in place */
        [data-testid="stSidebar"] {
            display: flex !important;
            transform: none !important;
            visibility: visible !important;
            min-width: 16rem !important;
            overflow-x: hidden !important; /* <--- ADD THIS LINE */
        }

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

        [data-testid="collapsedControl"] {
        }

        section[data-testid="stSidebar"], div[data-testid="stSidebar"] {
            min-width: 320px;
            max-width: 320px;
            width: 320px !important;
            position: fixed;
            left: 0;
            top: 0;
            height: 100vh;
            overflow-y: auto;
            border-right: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(9, 23, 39, 0.98), rgba(7, 17, 31, 0.98));
            z-index: 1000;
        }

        section.main {
            margin-left: 320px;
        }

        section[data-testid="stSidebar"] > div:first-child,
        div[data-testid="stSidebar"] > div:first-child {
            padding-top: 1.1rem;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
        div[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-bottom: 1.25rem;
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

        .sidebar-section {
            margin-top: 1rem;
        }

        .sidebar-row {
            margin-bottom: 0.55rem;
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

        .sidebar-status {
            display: inline-block;
            border: 1px solid rgba(57, 215, 170, 0.32);
            background: rgba(57, 215, 170, 0.10);
            color: #8df2d1;
            border-radius: 99px;
            padding: 0.28rem 0.6rem;
            font-size: 0.72rem;
            font-weight: 700;
        }

        .sidebar-muted {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.5;
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
# SIDEBAR: Context & System Status
# -----------------------------------------------------------------------------
with st.sidebar:
    
    # 1. New Plan Button
    if st.button("New plan", use_container_width=True, type="primary"):
        # Clear the chat history when starting a new plan
        st.session_state.conversation = {"messages": []} 
        st.rerun()

    st.divider()

    # 2. Active Group
    st.header("Workspace")
    st.selectbox(
        "Active group",
        options=["Hackathon Team", "Weekend Hike", "Dinner Squad"],
        index=0,
        key="active_group_selector"
    )
    
    st.divider()

    # 3. Saved Profiles (Dynamic Mapping)
    st.subheader("Saved Profiles")
    
    # Check if we have dynamic profiles loaded from your SQLite/JSON backend
    if "loaded_profiles" in st.session_state and st.session_state.loaded_profiles:
        for profile in st.session_state.loaded_profiles:
            # Assuming profile is a dict: {'name': 'Alice', 'transport': 'Transit', 'address': '123 St', 'notes': 'Vegan'}
            st.markdown(f"**{profile.get('name', 'Unknown')}**")
            st.caption(f"Transport: {profile.get('transport', 'N/A')}")
            st.caption(f"Address: {profile.get('address', 'N/A')}")
            
            # Conditionally render extra constraints if they exist
            if profile.get('notes'):
                st.caption(f"Notes: {profile.get('notes')}")
            
            st.write("") # Small spacer between profiles
    else:
        st.caption("No saved profiles for this group.")

    st.divider()

    # 4 & 5. Status Indicators
    st.subheader("System Status")
    st.caption("Local Memory: Active")
    st.caption("Local vLLM: Online")


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


def sync_group_profiles() -> None:
    """
    Load the visible profile cards for the active group.
    """
    st.session_state.loaded_profiles = load_group_profiles(st.session_state.group_id)


def render_profile_card(raw_profile: dict) -> None:
    """
    Render one compact Friend Profile card in the sidebar.
    """
    profile = normalize_profile(raw_profile)

    dietary = ", ".join(profile["dietary_restrictions"]) or "No restrictions"
    interests = ", ".join(profile["interests"]) or "Not specified"

    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-name">{profile["name"]}</div>
            <div class="profile-detail">
                {profile["address"]}<br>
                {profile["transit_mode"]} &nbsp; · &nbsp; {profile["budget"]}<br>
                {dietary}<br>
                {interests}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_profile_line(raw_profile: dict) -> None:
    """
    Render one compact constraint summary line for the sidebar.
    """
    profile = normalize_profile(raw_profile)
    dietary = ", ".join(profile["dietary_restrictions"]) or "No restrictions"
    transit = profile["transit_mode"]

    st.caption(f'{profile["name"]}: {dietary} · {transit}')


# -----------------------------------------------------------------------------
# Application Initialization
# -----------------------------------------------------------------------------
initialize_state()

if not st.session_state.loaded_profiles:
    sync_group_profiles()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1 class="hero-title">GatherPoint</h1>
        <p class="hero-subtitle">
            Local-first group coordination powered by AMD ROCm & vLLM.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Main Chat Workspace
# -----------------------------------------------------------------------------
st.markdown('<p class="section-label">Conversation</p>', unsafe_allow_html=True)

for message in st.session_state.conversation["messages"]:
    avatar = "🧑" if message["role"] == "user" else "📍"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask GatherPoint to plan a meetup...",
    key="chat_input",
)

if prompt:
    add_message(st.session_state.conversation, "user", prompt)

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📍"):
        status = st.status("GatherPoint is planning...", expanded=True)

        try:
            status.write("Loading Friend Profiles and saved group context...")
            status.write(
                "Preparing a bounded multi-turn prompt for local inference..."
            )
            status.write(
                "Checking available meetup tools and constraints..."
            )

            answer, profiles, memories = run_agent_turn(
                group_id=st.session_state.group_id,
                user_message=prompt,
                conversation=st.session_state.conversation,
            )

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

    add_message(st.session_state.conversation, "assistant", answer)