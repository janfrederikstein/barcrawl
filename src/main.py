"""
main.py — entry point

Invokes the agent with a user message and prints the response.
The agent loop (LLM → tools → LLM → … → answer) is defined in agent.py.
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from agent import app


def run_agent(user_message: str) -> str:
    """Send a single message to the agent and return its final response."""
    result = app.invoke({
        "messages": [HumanMessage(content=user_message)]
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    print("=== Agent starter ===\n")
    user_input = input("Ask the agent something: ").strip()

    print("\nThinking...\n" + "-" * 40)
    print(run_agent(user_input))
