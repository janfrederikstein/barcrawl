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


# ---------------------------------------------------------------------------
# 1. TOOLS
#    Tools are plain Python functions the LLM can choose to call.
#    Decorate them with @tool and write a clear docstring — the model reads
#    the docstring to understand when and how to use the tool.
#
#    For this project you'll want tools that:
#      - Search for bars near a location
#      - Look up details (hours, vibe, reviews) for a specific bar
#    The building blocks for both are already in src/tools/places.py.
# ---------------------------------------------------------------------------

@tool
def example_tool(query: str) -> str:
    """
    A placeholder tool. Replace this with something useful.

    Args:
        query: Whatever the model wants to pass in.
    """
    # TODO: replace with real logic, e.g. call search_bars_nearby()
    return f"[example_tool] received: {query}"


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

tools = [example_tool]
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
