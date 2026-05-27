import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from tools.places import get_bar_details, search_bars_nearby


# ---------------------------------------------------------------------------
# Tools — these are what the agent can call
# ---------------------------------------------------------------------------

@tool
def find_bars(location: str, radius_meters: int = 1000, preferences: str = "") -> str:
    """
    Find bars near a location. Returns a numbered list of nearby bars with
    name, address, rating, and place_id.

    Args:
        location: Address, neighborhood, or city to search near.
        radius_meters: Search radius in meters (default 1000).
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
            lines.append(f"   Price level: {'$' * (bar['price_level'] + 1)}")
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
        lines.append(f"Price level: {'$' * (details['price_level'] + 1)}")
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
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

tools = [find_bars, get_bar_info]
tool_node = ToolNode(tools)

model = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(tools)


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END


def call_model(state: AgentState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()
