from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from agent import app

SYSTEM_PROMPT = """You are a helpful and enthusiastic bar crawl planner.

Given a location and the user's preferences, your job is to:
1. Search for bars in the area using the find_bars tool
2. Get details on the most promising candidates using get_bar_info
3. Select 4–6 bars that best match the user's preferences
4. Arrange them in a logical geographic order (so the crawl makes sense to walk)
5. Present the itinerary clearly, with the bar name, address, a short description, and why it fits the vibe

Be specific and enthusiastic. Make the crawl sound fun!"""


def plan_bar_crawl(location: str, preferences: str) -> str:
    user_message = (
        f"I want to do a bar crawl in: {location}\n"
        f"My preferences: {preferences}\n\n"
        "Please plan a great bar crawl for me!"
    )

    result = app.invoke({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            HumanMessage(content=user_message),
        ]
    })

    return result["messages"][-1].content


if __name__ == "__main__":
    print("=== Bar Crawl Planner ===\n")
    location = input("Enter a location (address, neighborhood, or city): ").strip()
    preferences = input("What are you looking for? (e.g. 'craft beer, lively, not too expensive'): ").strip()

    print("\nPlanning your bar crawl — this may take a moment...\n")
    print("-" * 60)

    itinerary = plan_bar_crawl(location, preferences)
    print(itinerary)
