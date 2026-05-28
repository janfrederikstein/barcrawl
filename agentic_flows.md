# Agentic flows — the basics

## What is an agent?

A regular LLM call is a single round-trip: you send a prompt, you get a response, done.

An **agent** is different. It runs in a loop:

```
user message
     │
     ▼
  ┌───────┐        ┌───────┐
  │  LLM  │──────► │ tools │
  └───────┘        └───┬───┘
     ▲                 │
     └─────────────────┘
     (loop until no tool call)
     │
     ▼
 final answer
```

On each iteration the model either:
- **calls a tool** — the tool runs, the result is added to the conversation, and the model runs again with that new context, or  
- **responds directly** — the loop ends and that response is returned to the user.

This lets a single user message trigger multiple tool calls and reasoning steps before a final answer is produced.

---

## The three building blocks

### 1. State

State is the shared memory of the agent — a list of all messages so far (user input, model responses, and tool results). Every node in the graph reads from and writes to this list.

In this project:

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
```

`operator.add` means new messages are **appended**, not replaced, so nothing is ever lost mid-loop.

---

### 2. Nodes

A node is just a Python function that takes the current state and returns a dict of new messages to add. There are two nodes in this project:

| Node | What it does |
|------|-------------|
| `agent` | Sends the message history to the LLM and gets the next response |
| `tools` | Runs the tool the model requested and captures the output |

---

### 3. Edges

Edges connect nodes and control the flow. A **conditional edge** is the key to the agent loop:

```python
def should_continue(state):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END
```

After every LLM response we check: did the model ask to use a tool?
- **Yes** → route to the `tools` node, then back to `agent`
- **No** → route to `END` and return the answer

---

## How it maps to this project

The bar crawl planner works exactly like this:

1. The user asks for a bar crawl in a given area.
2. The `agent` node sends the request to the LLM.
3. The LLM calls `find_bars` to get a list of nearby bars.
4. The `tools` node runs `find_bars` and appends the results to state.
5. The LLM calls `get_bar_info` one or more times for promising candidates.
6. Once it has enough information, the LLM writes the final itinerary and the loop ends.

The model decides **how many tool calls to make and in what order** — that's the "agentic" part.

---

## Where to go next

- **`src/agent.py`** — the graph definition with inline comments on each piece
- **`src/tools/places.py`** — the Google Places API calls behind `find_bars` and `get_bar_info`
- **`solution` branch** — a complete working implementation you can diff against your own
