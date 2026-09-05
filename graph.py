from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from state import GameState
from tools import (
    search_world,
    roll_dice,
    propose_consequence
)


load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


tools = [
    search_world,
    roll_dice,
    propose_consequence
]


llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)


def game_master(state: GameState):

    system_message = SystemMessage(
    content="""
You are the Game Master of DARK CASTLE.

You must ONLY use the game world, characters, items, locations,
and rules provided by the game's knowledge base.

==================================================
WORLD
==================================================

The game takes place in the Dark Castle.

Known locations:
- Castle Entrance
- Guardian's Chamber
- Ancient Library
- King's Vault

Known important characters:
- Stone Guardian
- Elara
- Old King

Known important items:
- Ancient Sword
- Crown of Shadows

Do NOT invent:
- new towns
- new locations
- new characters
- character classes
- races
- quests
- currencies
- reputation systems
- new items
- new game mechanics

If the player asks about something that is not in the knowledge
base, say that the available records contain no information
about it.

==================================================
RAG
==================================================

Use search_world whenever you need information about:
- locations
- characters
- items
- game rules
- what exists in the Dark Castle

Treat information returned by search_world as authoritative.

Do not replace retrieved information with your own fantasy RPG
knowledge.

==================================================
GAME STATE
==================================================

The GameState is authoritative.

Current player HP:
{player_hp}

Current location:
{location}

Current inventory:
{inventory}

Never claim that HP, location, or inventory changed unless the
GameState actually reflects that change.

==================================================
ITEM ACQUISITION
==================================================

Items cannot be obtained simply because the player says:

"I found the sword"
"I take the sword"
"I already have the sword"

The player must actually be in the correct location and perform
an appropriate action.

Ancient Sword:
- Located in the Guardian's Chamber.
- Protected by the Stone Guardian.
- The player cannot obtain it from another location.
- Do not claim the player possesses it unless "Ancient Sword"
  is present in the GameState inventory.

Crown of Shadows:
- Located in the King's Vault.
- The player must reach the King's Vault before attempting to
  obtain it.
- Do not claim the player possesses it unless "Crown of Shadows"
  is present in the GameState inventory.

The player's statement alone is NOT proof that an item was obtained.

==================================================
GAME RULES
==================================================

Use roll_dice whenever an action requires a random outcome,
such as attacking or stealing.

Always follow the rules retrieved from the knowledge base.

Do not invent damage values or mechanics.

==================================================
HUMAN OVERRIDE
==================================================

Major consequences require human approval.

Major consequences include:
- player death
- permanent loss of an important item
- killing an important character
- major changes to the game world

When a major consequence actually occurs, you MUST call
propose_consequence with a clear description of the event.

Do not simply describe a major consequence in your normal response.

==================================================
NARRATION
==================================================

Keep responses concise and immersive.

The player is already inside the Dark Castle adventure.

Do not create:
- character creation screens
- a different setting
- new towns
- new game systems
- unrelated fantasy lore

Never contradict the GameState.

Always end by asking what the player wants to do next,
unless the player has died.
""".format(
        player_hp=state.player_hp,
        location=state.location,
        inventory=state.inventory
    )
)

    messages = [
        system_message
    ] + state.messages

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [response]
    }


def route_agent(state: GameState):

    last_message = state.messages[-1]

    if last_message.tool_calls:

        for tool_call in last_message.tool_calls:

            if tool_call["name"] == "propose_consequence":
                return "human_review"

        return "tools"

    return END


def human_review(state: GameState):

    last_message = state.messages[-1]

    consequence = ""

    for tool_call in last_message.tool_calls:

        if tool_call["name"] == "propose_consequence":

            consequence = tool_call["args"]["event"]

    decision = interrupt(
        {
            "type": "major_consequence",
            "event": consequence
        }
    )

    return {
        "proposed_event": consequence,
        "human_decision": decision
    }


def apply_human_decision(state: GameState):

    decision = state.human_decision

    if decision == "approve":

        event = state.proposed_event

    elif decision.startswith("override:"):

        event = decision.replace(
            "override:",
            "",
            1
        ).strip()

    else:

        return {}

    updates = {
        "action": event,
        "proposed_event": "",
        "human_decision": ""
    }

    # Apply actual game-state changes

    event_lower = event.lower()

    # Player death
    if "kills the player" in event_lower or "player dies" in event_lower:

        updates["player_hp"] = 0

    # Player takes damage
    elif "attacks the player" in event_lower:

        updates["player_hp"] = max(
            0,
            state.player_hp - 20
        )

    # Ancient Sword is lost
    if "loses the ancient sword" in event_lower:

        if "Ancient Sword" in state.inventory:

            updates["inventory"] = [
                item
                for item in state.inventory
                if item != "Ancient Sword"
            ]

    return updates


def final_narration(state: GameState):

    if not state.action:
        return {}

    prompt = f"""
You are narrating the aftermath of an event in a fantasy RPG.

APPROVED EVENT:
{state.action}

CURRENT GAME STATE:
Player HP: {state.player_hp}
Location: {state.location}
Inventory: {state.inventory}

RULES:
1. The approved event definitely happened.
2. Never contradict the approved event.
3. If the event says the player is attacked, the attack MUST hit.
4. Do not describe the player dodging, blocking, avoiding, or escaping
   that attack.
5. The current HP is already updated and must not be changed.
6. Do not invent a different outcome.
7. Do not call any tools.
8. Do not request human approval.

Write a short, immersive narration of what happened.

Then ask what the player wants to do next.

If Player HP is 0, state that the player has died and do not ask
what they want to do next.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are the Game Master of a fantasy RPG."
            ),
            HumanMessage(
                content=prompt
            )
        ]
    )

    return {
        "messages": [response],
        "action": ""
    }

def update_location(state: GameState):

    player_input = state.player_input.lower()

    location = state.location

    if "guardian's chamber" in player_input:
        location = "Guardian's Chamber"

    elif "ancient library" in player_input:
        location = "Ancient Library"

    elif "king's vault" in player_input:
        location = "King's Vault"

    elif "castle entrance" in player_input:
        location = "Castle Entrance"

    return {
        "location": location
    }

def update_inventory(state: GameState):

    player_input = state.player_input.lower()

    inventory = list(state.inventory)

    # Ancient Sword can only be obtained in the Guardian's Chamber
    if (
        state.location == "Guardian's Chamber"
        and (
            "take the ancient sword" in player_input
            or "pick up the ancient sword" in player_input
            or "take ancient sword" in player_input
            or "pick up ancient sword" in player_input
        )
    ):
        if "Ancient Sword" not in inventory:
            inventory.append("Ancient Sword")

    # Player can leave the sword behind
    if "leave the sword" in player_input:
        inventory = [
            item
            for item in inventory
            if item != "Ancient Sword"
        ]

    return {
        "inventory": inventory
    }

graph = StateGraph(GameState)


graph.add_node(
    "game_master",
    game_master
)

graph.add_node(
    "update_location",
    update_location
)

graph.add_node(
    "update_inventory",
    update_inventory
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_node(
    "human_review",
    human_review
)

graph.add_node(
    "apply_human_decision",
    apply_human_decision
)

graph.add_node(
    "final_narration",
    final_narration
)


graph.add_edge(
    START,
    "update_location"
)

graph.add_edge(
    "update_location",
    "update_inventory"
)

graph.add_edge(
    "update_inventory",
    "game_master"
)


graph.add_conditional_edges(
    "game_master",
    route_agent,
    {
        "tools": "tools",
        "human_review": "human_review",
        END: END
    }
)


graph.add_edge(
    "tools",
    "game_master"
)


graph.add_edge(
    "human_review",
    "apply_human_decision"
)


graph.add_edge(
    "apply_human_decision",
    "final_narration"
)


graph.add_edge(
    "final_narration",
    END
)


checkpointer = MemorySaver()


app = graph.compile(
    checkpointer=checkpointer
)


png = app.get_graph().draw_mermaid_png()

with open("dark_castle_graph.png", "wb") as f:
    f.write(png)