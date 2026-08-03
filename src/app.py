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
        :root {
            --bg: #ffffff;
            --sidebar-bg: #fdfdfd;
            --line: #f2f2f2;
            --text: #111322;
            --muted: #767583;
            --primary: #0e34dd;
            --chip-bg: #edf0f7;
            --input-bg: #ffffff;
            --message-bg: #f9fafb;
            --accent: #d44928;
            --accent-hover: #bf4022;
            --accent-border: #ab3d24;
            --input-border: #e5e7eb;
            --message-border: #eae9ee;
            --label-color: #414651;
        }

        .stApp {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: var(--bg);
            color: var(--text);
        }

        /* Hide Streamlit's default application chrome for a cleaner demo UI. */
        #MainMenu, footer, header {
            visibility: hidden;
        }

        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 6rem;
            margin-left: 0;
        }

        section[data-testid="stSidebar"],
        div[data-testid="stSidebar"] {
            display: block !important;
            width: 242px !important;
            min-width: 242px !important;
            max-width: 242px !important;
            position: fixed !important;
            left: 0 !important;
            top: 0 !important;
            bottom: 0 !important;
            height: 100vh !important;
            overflow-y: auto !important;
            border-right: 1px solid var(--line) !important;
            background: var(--sidebar-bg) !important;
            box-shadow: none !important;
            overflow-x: hidden !important;
            z-index: 999999 !important;
            transform: none !important;
        }

        section[data-testid="stSidebar"] > div:first-child,
        div[data-testid="stSidebar"] > div:first-child {
            padding-top: 0.8rem;
            padding-left: 0.7rem;
            padding-right: 0.7rem;
            background: var(--sidebar-bg) !important;
            height: 100% !important;
        }

        section.main .block-container {
            margin-left: 242px !important;
        }

        h1, h2, h3, label, p, span {
            color: var(--text);
        }

        h1 {
            font-size: 2.25rem;
            font-weight: 700;
            line-height: 1.2;
        }

        .hero {
            border-bottom: 1px solid var(--line);
            border-radius: 0;
            background: var(--bg);
            padding: 0 0 1.25rem 0;
            margin-bottom: 1.25rem;
            box-shadow: none;
        }

        .hero-title {
            color: var(--text);
            font-size: 2.6rem;
            font-weight: 700;
            line-height: 1.15;
            margin: 0;
        }

        .hero-subtitle {
            color: var(--muted);
            margin-top: 0.45rem;
            margin-bottom: 0;
            line-height: 1.45;
            font-size: 0.95rem;
        }

        .section-label {
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0.3rem 0 0.6rem 0;
        }

        .sidebar-section {
            margin-top: 1rem;
        }

        [data-testid="stSidebar"] .stCaption {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.5;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] .stMarkdown p {
            color: var(--text) !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] h2 {
            font-size: 1.05rem;
            margin-top: 0.15rem;
            margin-bottom: 0.35rem;
        }

        [data-testid="stSidebar"] h3 {
            font-size: 0.92rem;
            margin-top: 0.75rem;
            margin-bottom: 0.2rem;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
        [data-testid="stSidebar"] [data-testid="stTextInput"] label {
            color: var(--label-color);
            font-size: 0.78rem;
            font-weight: 600;
        }

        /* Style user and assistant messages as product chat cards. */
        [data-testid="stChatMessage"] {
            border: 1px solid var(--message-border);
            background: var(--message-bg);
            border-radius: 8px;
            margin-bottom: 0.65rem;
            padding: 0.25rem 0.4rem;
            box-shadow: none;
        }

        [data-testid="stChatInput"] {
            border: 1px solid var(--message-border);
            border-radius: 8px;
            background: var(--input-bg);
            box-shadow: 0 1px 1px rgba(16, 24, 40, 0.04);
        }

        div.stButton > button {
            border-radius: 6px;
            border: 1px solid var(--accent-border);
            background: var(--accent);
            color: var(--bg);
            font-weight: 600;
            font-size: 0.8rem;
            box-shadow: none;
        }

        div.stButton > button:hover {
            background: var(--accent-hover);
            border-color: #99351f;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-radius: 6px;
            border-color: var(--input-border);
            background: var(--bg);
            color: var(--text);
        }

        /* Chat input bar text must be white against its dark background. */
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInputTextArea"] {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        /* Keep the bottom chat input from overlapping the fixed sidebar. */
        [data-testid="stBottom"] {
            left: 16rem !important;
            width: calc(100% - 16rem) !important;
            z-index: 99 !important;
            background-color: transparent !important;
        }

        /* --- AGENT STATUS WIDGET (st.status renders as an expander): STATIC WHITE BUBBLE --- */
        /* Force a permanent white card look and remove every hover/focus transition. */
        [data-testid="stExpander"],
        [data-testid="stExpander"] *,
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary:hover,
        [data-testid="stExpander"] summary:active,
        [data-testid="stExpander"] summary:focus,
        [data-testid="stExpanderDetails"] {
            background-color: #ffffff !important;
            background: #ffffff !important;
            border-color: var(--message-border) !important;
            box-shadow: none !important;
            transition: none !important;
        }

        /* Force every text node inside the status widget to stay dark. */
        [data-testid="stExpander"] *,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] span,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] div {
            color: #111322 !important;
            font-weight: 500 !important;
        }

        /* Force the loading spinner / icons to be dark at all times. */
        [data-testid="stExpander"] svg,
        [data-testid="stExpander"] svg * {
            color: #111322 !important;
            fill: #111322 !important;
            stroke: #111322 !important;
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


def sync_group_profiles() -> None:
    """
    Load the visible profile cards for the active group.
    """
    st.session_state.loaded_profiles = load_group_profiles(st.session_state.group_id)


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
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Chatform")

    if st.button("New plan", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.divider()
    st.subheader("Workspace")

    group_candidates = [
        st.session_state.group_id,
        "hackathon-team",
        "weekend-hike",
        "dinner-squad",
    ]
    group_options = list(dict.fromkeys(group_candidates))
    active_group = st.selectbox("Active group", options=group_options, index=0)

    if active_group != st.session_state.group_id:
        st.session_state.group_id = active_group
        sync_group_profiles()

    st.divider()
    st.subheader("Saved Profiles")

    if st.session_state.loaded_profiles:
        for raw_profile in st.session_state.loaded_profiles:
            render_sidebar_profile_line(raw_profile)
    else:
        st.caption("No saved profiles for this group.")

    st.divider()
    st.subheader("Local Memory")
    st.caption("Memory: Active")

    st.divider()
    st.subheader("System Status")
    st.caption("Local vLLM: Online")


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <p class="hero-subtitle">Home</p>
        <h1 class="hero-title">Welcome to GatherPoint</h1>
        <p class="hero-subtitle">
            Local-first group coordination powered by AMD ROCm and vLLM.
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
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask GatherPoint to plan a meetup...",
    key="chat_input",
)

if prompt:
    add_message(st.session_state.conversation, "user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
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