import random

from langchain_core.tools import tool

from rag import create_retriever


retriever = create_retriever()


@tool
def search_world(query: str) -> str:
    """Search the game world for locations, characters, items and rules."""

    docs = retriever.invoke(query)

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


@tool
def roll_dice(request: str = "") -> int:
    """Roll a twenty-sided dice. The request argument can be left empty."""

    return random.randint(1, 20)


@tool
def propose_consequence(event: str) -> str:
    """Propose a major consequence that requires human approval."""

    return event