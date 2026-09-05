from typing import Annotated

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GameState(BaseModel):

    player_input: str = ""

    location: str = "Castle Entrance"
    player_hp: int = 100
    inventory: list[str] = Field(default_factory=list)

    context: list[str] = Field(default_factory=list)

    action: str = ""
    dice_roll: int = 0

    proposed_event: str = ""
    major_consequence: bool = False
    human_decision: str = ""

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ] = Field(default_factory=list)