import uuid

import streamlit as st

from graph import app
from state import GameState

from langchain_core.messages import HumanMessage
from langgraph.types import Command


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Dark Castle",
    page_icon="♜",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom styling (without modifying Streamlit sidebar positioning/width)
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* ===== Global ===== */

    .stApp {
        background: #11110f;
        color: #ded6c6;
    }

    /* Hide Streamlit branding while keeping header functional for sidebar toggle */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }


    /* ===== Typography ===== */

    .castle-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 34px;
        letter-spacing: 5px;
        color: #ded6c6;
        margin: 0 0 2px 0;
        line-height: 1.2;
    }

    .castle-subtitle {
        color: #817b70;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 12px;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    .divider {
        border: none;
        border-top: 1px solid #302f2a;
        margin: 20px 0 28px 0;
    }


    /* ===== Sidebar styling (colors & theme only) ===== */

    section[data-testid="stSidebar"] {
        background: #151512;
        border-right: 1px solid #2f2e29;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #ded6c6;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #302f2a;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: 1px solid #4a463b;
        color: #817b70;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 12px;
        letter-spacing: 1px;
        padding: 8px 16px;
        transition: all 0.2s ease;
        width: 100%;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #8b7652;
        color: #ded6c6;
    }


    /* ===== Journal ===== */

    .journal {
        background: #171714;
        border: 1px solid #302f2a;
        border-radius: 2px;
        padding: 32px 36px;
        min-height: 520px;
    }

    .journal-label {
        color: #6b6559;
        font-size: 10px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 24px;
        font-family: Georgia, "Times New Roman", serif;
    }


    /* ===== Player / GM Messages ===== */

    .player-message {
        border-left: 2px solid #8b7652;
        padding: 12px 18px;
        margin: 0 0 28px 0;
        color: #cfc7b8;
        background: rgba(139, 118, 82, 0.04);
    }

    .gm-message {
        color: #bdb6a8;
        line-height: 1.8;
        font-family: Georgia, "Times New Roman", serif;
        margin: 0 0 32px 0;
        padding-left: 2px;
    }

    .message-label {
        color: #70695e;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 9px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }


    /* ===== Status (sidebar) ===== */

    .status-label {
        color: #6b6559;
        font-size: 9px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        font-family: Georgia, "Times New Roman", serif;
    }

    .status-value {
        color: #d5cebf;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 16px;
        margin-top: 4px;
    }


    /* ===== Map ===== */

    .map-container {
        background: #171714;
        border: 1px solid #302f2a;
        border-radius: 2px;
        padding: 12px;
    }

    .map-label {
        color: #6b6559;
        font-size: 10px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
        font-family: Georgia, "Times New Roman", serif;
    }


    /* ===== HITL / Fate ===== */

    .fate-box {
        background: #1b1915;
        border: 1px solid #66563c;
        border-radius: 2px;
        padding: 28px 32px;
        margin: 28px 0;
    }

    .fate-title {
        color: #b39a6c;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 18px;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }

    .fate-event {
        color: #c8c0b2;
        line-height: 1.7;
        font-family: Georgia, "Times New Roman", serif;
        margin-bottom: 24px;
    }

    .stButton > button[data-testid="stBaseButton-secondary"],
    .fate-box .stButton > button {
        background: transparent;
        border: 1px solid #66563c;
        color: #b39a6c;
        font-family: Georgia, "Times New Roman", serif;
        letter-spacing: 1px;
        transition: all 0.2s ease;
    }

    .fate-box .stButton > button:hover {
        background: rgba(102, 86, 60, 0.15);
        border-color: #8b7652;
        color: #ded6c6;
    }


    /* ===== Chat input ===== */

    .stChatInputContainer {
        background: #11110f !important;
        border-top: 1px solid #302f2a !important;
    }

    .stChatInputContainer textarea {
        background: #191916 !important;
        color: #ded6c6 !important;
        font-family: Georgia, "Times New Roman", serif;
    }

    .stChatInputContainer textarea::placeholder {
        color: #6b6559 !important;
    }


    /* ===== Welcome message ===== */

    .welcome-text {
        color: #bdb6a8;
        font-family: Georgia, "Times New Roman", serif;
        line-height: 1.8;
        margin-bottom: 24px;
    }


    /* ===== Scrollbar (subtle) ===== */

    section[data-testid="stSidebar"] ::-webkit-scrollbar {
        width: 4px;
    }

    section[data-testid="stSidebar"] ::-webkit-scrollbar-track {
        background: transparent;
    }

    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
        background: #302f2a;
        border-radius: 2px;
    }


    /* ===== Responsive: narrow screens ===== */

    @media (max-width: 900px) {
        .castle-title {
            font-size: 26px;
            letter-spacing: 3px;
        }

        .journal {
            padding: 20px 24px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Game initialization
# --------------------------------------------------

if "state" not in st.session_state:
    st.session_state.state = GameState()
    st.session_state.config = {
        "configurable": {"thread_id": str(uuid.uuid4())}
    }
    st.session_state.history = []
    st.session_state.pending_interrupt = None

state = st.session_state.state
config = st.session_state.config


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("DARK CASTLE")
    st.write("SIDEBAR TEST")

    st.markdown("---")

    # Vitality / HP

    st.markdown(
        '<div class="status-label">Vitality</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="status-value">{state.player_hp} / 100</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Location

    st.markdown(
        '<div class="status-label">Current Location</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="status-value">{state.location}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Inventory

    st.markdown(
        '<div class="status-label">Possessions</div>',
        unsafe_allow_html=True,
    )

    if state.inventory:
        for item in state.inventory:
            st.markdown(
                f'<div class="status-value" style="font-size:13px;">{item}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="color:#625e55; font-family: Georgia, serif;">'
            "Nothing carried</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Begin Again

    if st.button("Begin Again", use_container_width=True):
        st.session_state.state = GameState()
        st.session_state.config = {
            "configurable": {"thread_id": str(uuid.uuid4())}
        }
        st.session_state.history = []
        st.session_state.pending_interrupt = None
        st.rerun()




# --------------------------------------------------
# Main game layout: Journal + Map
# --------------------------------------------------

col_journal, col_map = st.columns([3, 1], gap="medium")


# --------------------------------------------------
# Journal
# --------------------------------------------------

with col_journal:

    # Header

    st.markdown(
        '<div class="castle-title">DARK CASTLE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="castle-subtitle">A Chronicle of Uncertain Fate</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="journal-label">The Journal</div>',
        unsafe_allow_html=True,
    )

    # Conversation history

    if not st.session_state.history:
        st.markdown(
            '<div class="welcome-text">'
            "You stand before the Dark Castle. "
            "Its ancient stones loom against a sky choked with ash. "
            "The gate yawns open, as though expecting you.</div>",
            unsafe_allow_html=True,
        )
    else:
        for message in st.session_state.history:
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div class="player-message">
                        <div class="message-label">You</div>
                        {message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="gm-message">
                        <div class="message-label">The Chronicle</div>
                        {message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# --------------------------------------------------
# Map
# --------------------------------------------------

with col_map:

    st.markdown(
        '<div class="map-label">The Dark Castle</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="map-container">',
        unsafe_allow_html=True,
    )

    st.image(
        "assets/dark_castle_map.png",
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="status-label">Current Location</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="status-value" style="font-size:14px;">'
        f"{state.location}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Human-in-the-Loop review
# --------------------------------------------------

if st.session_state.pending_interrupt:

    interrupt_data = st.session_state.pending_interrupt

    st.markdown(
        """
        <div class="fate-box">
            <div class="fate-title">The Fate of This Moment</div>
            <div class="fate-event">
        """,
        unsafe_allow_html=True,
    )

    st.write(interrupt_data["event"])

    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Accept consequence

    with col1:
        if st.button("Accept Fate", use_container_width=True):
            result = app.invoke(
                Command(resume="approve"),
                config=config,
            )
            st.session_state.pending_interrupt = None
            st.session_state.state = GameState(**result)
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": result["messages"][-1].content,
                }
            )
            st.rerun()

    # Override consequence

    with col2:
        override = st.text_input(
            "Alter the outcome",
            label_visibility="collapsed",
            placeholder="Describe another outcome...",
        )

        if st.button("Alter Fate", use_container_width=True):
            if override.strip():
                result = app.invoke(
                    Command(resume=f"override: {override}"),
                    config=config,
                )
                st.session_state.pending_interrupt = None
                st.session_state.state = GameState(**result)
                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": result["messages"][-1].content,
                    }
                )
                st.rerun()


# --------------------------------------------------
# Player input
# --------------------------------------------------

if not st.session_state.pending_interrupt:

    player_input = st.chat_input("What do you do?")

    if player_input:
        # Add player message to journal

        st.session_state.history.append(
            {"role": "user", "content": player_input}
        )

        # Update game state

        state.player_input = player_input
        state.messages.append(HumanMessage(content=player_input))

        # Run LangGraph

        result = app.invoke(state, config=config)

        # Human approval required

        if "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0]
            st.session_state.pending_interrupt = {
                "event": interrupt_data.value["event"]
            }
            st.rerun()

        # Normal response

        st.session_state.state = GameState(**result)
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": result["messages"][-1].content,
            }
        )
        st.rerun()
