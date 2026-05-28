"""
agent.py — Agentic flow starter

An "agentic flow" is a loop where an LLM:
  1. Receives the current conversation state (all messages so far)
  2. Decides whether to call a tool or give a final answer
  3. If it calls a tool → the tool runs, its output is added to the state
  4. The LLM sees the tool output and decides again  (back to step 1)
  5. When the LLM produces a final answer (no tool call) → the loop ends

The three building blocks in this file are:
  - State  : the shared conversation history
  - Nodes  : functions that read state and produce new messages
  - Edges  : rules that decide which node runs next

Tip: the solution branch has a fully-working bar crawl planner built
on exactly this skeleton — check it out once you've had a go yourself.
"""

import operator
import os
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from tools.places import get_bar_details, search_bars_nearby


# ---------------------------------------------------------------------------
# 1. TOOLS
#    Tools are plain Python functions the LLM can choose to call.
#    Decorate them with @tool and write a clear docstring — the model reads
#    the docstring to understand when and how to use the tool.
#
#    Two tools are provided here, both backed by the Google Places API
#    (see src/tools/places.py — you don't need to touch that file):
#      - find_bars   : search for bars near a location string
#      - get_bar_info: fetch details for a specific bar by its Place ID
#
#    Notice the pattern:
#      1. Call the Places API helper to get raw data
#      2. Format the result as a readable string
#      3. Return that string — the LLM reads it like text in the conversation
# ---------------------------------------------------------------------------

# Price levels come back from the Places API as string enums, not integers.
# This helper converts them to the familiar $ / $$ / $$$ notation.
_PRICE_LEVEL = {
    "PRICE_LEVEL_FREE": "Free",
    "PRICE_LEVEL_INEXPENSIVE": "$",
    "PRICE_LEVEL_MODERATE": "$$",
    "PRICE_LEVEL_EXPENSIVE": "$$$",
    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
}

def _fmt_price(level) -> str:
    if level is None:
        return ""
    if isinstance(level, str):
        return _PRICE_LEVEL.get(level, level)
    return "$" * (level + 1)  # fallback for legacy integer values


@tool
def find_bars(location: str, radius_meters: int = 1000, preferences: str = "") -> str:
    """
    Find bars near a location. Returns a numbered list of nearby bars with
    name, address, rating, and place_id.

    Args:
        location: Address, neighbourhood, or city to search near.
        radius_meters: Search radius in metres (default 1000).
        preferences: Optional keyword to filter bars (e.g. "craft beer", "cocktail").
    """
    bars = search_bars_nearby(location, radius_meters, keyword=preferences or None)
    if not bars:
        return f"No bars found near '{location}'. Try a broader search or larger radius."

    lines = [f"Found {len(bars)} bars near {location}:\n"]
    for i, bar in enumerate(bars, 1):
        lines.append(f"{i}. {bar['name']}")
        lines.append(f"   Address: {bar['address']}")
        if bar["rating"]:
            lines.append(f"   Rating: {bar['rating']} ({bar['user_ratings_total']} reviews)")
        if bar["price_level"] is not None:
            lines.append(f"   Price level: {_fmt_price(bar['price_level'])}")
        if bar["open_now"] is not None:
            lines.append(f"   Open now: {'Yes' if bar['open_now'] else 'No'}")
        lines.append(f"   Place ID: {bar['place_id']}")
        lines.append("")

    return "\n".join(lines)


@tool
def get_bar_info(place_id: str) -> str:
    """
    Get detailed information about a specific bar using its Google Place ID.
    Returns address, opening hours, rating, website, and a short description.

    Args:
        place_id: The Google Place ID of the bar (from find_bars results).
    """
    details = get_bar_details(place_id)

    lines = [f"**{details['name']}**"]
    if details["address"]:
        lines.append(f"Address: {details['address']}")
    if details["summary"]:
        lines.append(f"About: {details['summary']}")
    if details["rating"]:
        lines.append(f"Rating: {details['rating']} ({details.get('user_ratings_total', '?')} reviews)")
    if details["price_level"] is not None:
        lines.append(f"Price level: {_fmt_price(details['price_level'])}")
    if details["opening_hours"]:
        lines.append("Hours:\n" + "\n".join(f"  {h}" for h in details["opening_hours"]))
    if details["website"]:
        lines.append(f"Website: {details['website']}")
    if details["reviews"]:
        lines.append("Recent reviews:")
        for r in details["reviews"]:
            lines.append(f"  [{r['rating']}★] {r['text'][:120]}...")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. STATE
#    AgentState is a typed dict with a single key: `messages`.
#    The `operator.add` annotation means new messages are *appended*,
#    not replaced — so every node sees the full conversation history.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# ---------------------------------------------------------------------------
# 3. THE MODEL
#    ChatOpenAI pointed at the Solita LiteLLM endpoint.
#    .bind_tools() tells the model which tools it is allowed to call —
#    it will include their schemas in every request automatically.
# ---------------------------------------------------------------------------

tools = [find_bars, get_bar_info]
tool_node = ToolNode(tools)  # LangGraph helper: runs whichever tool the model picked

model = ChatOpenAI(
    model=os.environ.get("LITELLM_MODEL", "google/gemini-2.5-flash"),
    api_key=os.environ.get("LITELLM_API_KEY"),
    base_url=os.environ.get("LITELLM_BASE_URL", "https://app-litellmsn66ka.azurewebsites.net/v1"),
).bind_tools(tools)


# ---------------------------------------------------------------------------
# 4. NODES
#    A node is a function: AgentState → dict of new messages to append.
#    This graph has two nodes:
#      "agent" — calls the LLM and gets its next response
#      "tools" — runs whichever tool the model requested
# ---------------------------------------------------------------------------

def call_model(state: AgentState) -> dict:
    """Send the current message history to the LLM and return its response."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 5. GRAPH (the flow)
#    Nodes are wired together with edges.
#    A *conditional* edge lets us branch at runtime:
#      did the model's last message contain a tool call?
#        yes → go to "tools"
#        no  → END (the model's answer is the final output)
#    After a tool runs we always go back to "agent" so the model can
#    see the result and decide what to do next.
#
#    The resulting loop looks like:
#
#       ┌─────────┐
#       │  agent  │◄──────────────┐
#       └────┬────┘               │
#            │ tool call?         │
#       yes ─┤                    │
#            ▼                    │
#       ┌─────────┐               │
#       │  tools  │───────────────┘
#       └─────────┘
#            │ no tool call
#            ▼
#           END
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    """Route to 'tools' if the model wants to call one, otherwise finish."""
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END


graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()
