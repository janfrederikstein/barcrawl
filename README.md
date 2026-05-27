# Bar Crawl Planner — Hackathon Starter Kit

Your challenge is to build an agentic AI app that plans a personalized bar crawl based on a user's location and preferences.

---

## The Challenge

### Background

Planning a bar crawl is harder than it looks. You need to find bars that match the group's vibe, check which ones are actually open, sort them geographically so you're not zigzagging across town, and balance variety with personal taste.

AI agents are well-suited to this kind of task: they can search for information, reason about trade-offs, and produce a structured output — all in a loop until the job is done.

### Input

The user provides:
- A **location** — address, neighborhood, or city (e.g. `"Södermalm, Stockholm"` or `"Shoreditch, London"`)
- **Preferences** — free-text description of what they're looking for (e.g. `"craft beer, lively atmosphere, not too pricey, dog-friendly"`)

### Output

A bar crawl plan with:
- **4–6 bar stops**, ordered so the route makes geographic sense
- For each bar: name, address, a short description, and why it fits the user's preferences
- (Bonus) Estimated walking time between stops, opening hours, price level

### Requirements

**Must Have**
- [ ] Accepts location and preference input from the user
- [ ] Uses at least one **agentic loop** — the agent decides what to look up based on what it finds
- [ ] Returns a coherent itinerary with 4–6 stops in a sensible order

**Nice to Have**
- [ ] A web or chat interface instead of a CLI
- [ ] Walking directions or a map between stops
- [ ] Handles edge cases gracefully (no bars found, all closed, conflicting preferences)
- [ ] Conversational refinement — the user can say "swap bar 3 for something quieter"
- [ ] Explanation of *why* each bar was chosen

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/janfstein93/barcrawl.git
cd barcrawl
```

### 2. Set up a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get your API keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

#### LiteLLM (Solitaire API key)
https://insider.solita.fi/sites/generative-ai/news/28723/launching-litellm-api-keys-for-competence-development-and-access-to-mcp-servers-in-github-copilot

#### Google Places API key

The starter uses the **Places API (New)** — Google's current generation API for searching and retrieving place data.

1. Create a Google Cloud account at [console.cloud.google.com](https://console.cloud.google.com). **New accounts get $300 in free credits**, more than enough for this hackathon.
2. Create a new project.
3. Go to **APIs & Services → Library** and enable **Places API (New)**.
4. Go to **APIs & Services → Credentials → Create credentials → API key** and paste the key into `.env`.

> **Troubleshooting:** Make sure you enable **Places API (New)** specifically — there is also a legacy "Places API" in the library which will not work with the starter code.

> **No API?** If you run into issues, the `data/` folder has pre-fetched fallback data — see [Fallback Data](#fallback-data) below.

### 4. Run the starter app

```bash
cd src
python main.py
```

You'll be prompted for a location and your bar preferences. The agent will search for nearby bars and return a planned crawl itinerary.

---

## Project Structure

```
barcrawl/
├── data/
│   ├── fallback_sveavagen.csv    # 100 pre-fetched bars near Sveavägen
│   ├── fallback_gotgatan.csv     # 100 pre-fetched bars near Götgatan
│   └── fallback_fridhemsplan.csv # 100 pre-fetched bars near Fridhemsplan
├── src/
│   ├── main.py                   # CLI entry point — start here
│   ├── agent.py                  # LangGraph agent with tools wired up
│   └── tools/
│       └── places.py             # Google Places API (New) helpers
├── .env.example                  # API key template
└── requirements.txt
```

## What's Pre-built for You

**`src/tools/places.py`** — three ready-to-use functions:

| Function | Description |
|----------|-------------|
| `search_bars_nearby(location, radius_meters, keyword)` | Search for bars near a location string |
| `search_bars_by_text(query, lat_lng, radius_meters, max_results)` | Paginated free-text search — good for specific queries like `"craft beer bar"` |
| `get_bar_details(place_id)` | Full details for a specific bar: hours, reviews, website, price level |

**`src/agent.py`** — a LangGraph ReAct agent with `find_bars` and `get_bar_info` tools already registered.

**`src/main.py`** — a working CLI that takes location + preferences and prints an itinerary.

## Fallback Data

If you can't get the Google Places API working, the `data/` folder contains pre-fetched CSV files with 100 bars each for three Stockholm locations:

| File | Location |
|------|----------|
| `data/fallback_sveavagen.csv` | Bars near Sveavägen, Stockholm |
| `data/fallback_gotgatan.csv` | Bars near Götgatan, Stockholm |
| `data/fallback_fridhemsplan.csv` | Bars near Fridhemsplan, Stockholm |

Each row contains: `name`, `category`, `address`, `lat`, `lng`, `rating`, `user_ratings_total`, `price_level`, `summary`, `phone`, `website`, `opening_hours`, `place_id`.

```python
import pandas as pd
df = pd.read_csv("../data/fallback_sveavagen.csv")
bars = df.to_dict(orient="records")
```

## Useful Resources

- [Anthropic docs](https://docs.anthropic.com)
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [Places API (New) reference](https://developers.google.com/maps/documentation/places/web-service/op-overview)

Good luck!
