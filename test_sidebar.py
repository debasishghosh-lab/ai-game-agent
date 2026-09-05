import streamlit as st

st.set_page_config(
    page_title="Sidebar Test",
    page_icon="?",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.write("Main content area")

with st.sidebar:
    st.title("SIDEBAR TEST")
    st.write("This should be visible in the sidebar")
    st.button("Test Button")
