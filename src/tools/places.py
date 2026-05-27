import os
import time
from typing import Optional

import requests

_BASE_URL = "https://places.googleapis.com/v1"


def _api_key() -> str:
    key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not key:
        raise ValueError("GOOGLE_PLACES_API_KEY is not set in your .env file")
    return key


def _headers(field_mask: str) -> dict:
    return {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": _api_key(),
        "X-Goog-FieldMask": field_mask,
    }


def geocode_location(location: str) -> tuple[float, float] | None:
    """Convert a location string to (lat, lng) using Places Text Search."""
    response = requests.post(
        f"{_BASE_URL}/places:searchText",
        headers=_headers("places.location,places.displayName"),
        json={"textQuery": location},
    )
    response.raise_for_status()
    places = response.json().get("places", [])
    if not places:
        return None
    loc = places[0]["location"]
    return loc["latitude"], loc["longitude"]


def search_bars_nearby_by_coords(
    lat_lng: tuple[float, float],
    radius_meters: int = 1000,
    keyword: Optional[str] = None,
) -> list[dict]:
    """Search for bars near a (lat, lng) coordinate pair."""
    body: dict = {
        "includedTypes": ["bar"],
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat_lng[0], "longitude": lat_lng[1]},
                "radius": float(radius_meters),
            }
        },
    }
    if keyword:
        body["textQuery"] = keyword

    field_mask = ",".join([
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.rating",
        "places.userRatingCount",
        "places.priceLevel",
        "places.currentOpeningHours",
        "places.location",
    ])

    response = requests.post(
        f"{_BASE_URL}/places:searchNearby",
        headers=_headers(field_mask),
        json=body,
    )
    response.raise_for_status()

    bars = []
    for place in response.json().get("places", []):
        bars.append({
            "name": place.get("displayName", {}).get("text"),
            "place_id": place.get("id"),
            "address": place.get("formattedAddress"),
            "rating": place.get("rating"),
            "user_ratings_total": place.get("userRatingCount"),
            "price_level": place.get("priceLevel"),
            "open_now": place.get("currentOpeningHours", {}).get("openNow"),
            "location": {
                "lat": place.get("location", {}).get("latitude"),
                "lng": place.get("location", {}).get("longitude"),
            },
        })

    return bars


def search_bars_nearby(
    location: str,
    radius_meters: int = 1000,
    keyword: Optional[str] = None,
) -> list[dict]:
    """
    Search for bars near a location string (address, neighborhood, or city).

    Returns a list of bars, each with: name, place_id, address, rating,
    user_ratings_total, price_level, open_now, and lat/lng coordinates.
    """
    coords = geocode_location(location)
    if not coords:
        return []
    return search_bars_nearby_by_coords(coords, radius_meters, keyword)


def search_bars_by_text(
    query: str,
    lat_lng: tuple[float, float],
    radius_meters: int = 2000,
    max_results: int = 60,
) -> list[dict]:
    """
    Search for bars using a free-text query with automatic pagination.

    Useful for specific searches like "craft beer bar" or "rooftop cocktail bar".
    Returns up to max_results bars, each with: name, place_id, address, rating,
    user_ratings_total, price_level, open_now, and lat/lng coordinates.
    """
    field_mask = ",".join([
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.rating",
        "places.userRatingCount",
        "places.priceLevel",
        "places.currentOpeningHours",
        "places.location",
    ])

    bars: dict[str, dict] = {}
    page_token: str | None = None

    while len(bars) < max_results:
        body: dict = {
            "textQuery": query,
            "maxResultCount": 20,
            "locationBias": {
                "circle": {
                    "center": {"latitude": lat_lng[0], "longitude": lat_lng[1]},
                    "radius": float(radius_meters),
                }
            },
        }
        if page_token:
            body["pageToken"] = page_token

        response = requests.post(
            f"{_BASE_URL}/places:searchText",
            headers=_headers(field_mask),
            json=body,
        )
        response.raise_for_status()
        data = response.json()

        for place in data.get("places", []):
            place_id = place.get("id")
            if place_id and place_id not in bars:
                bars[place_id] = {
                    "name": place.get("displayName", {}).get("text"),
                    "place_id": place_id,
                    "address": place.get("formattedAddress"),
                    "rating": place.get("rating"),
                    "user_ratings_total": place.get("userRatingCount"),
                    "price_level": place.get("priceLevel"),
                    "open_now": place.get("currentOpeningHours", {}).get("openNow"),
                    "location": {
                        "lat": place.get("location", {}).get("latitude"),
                        "lng": place.get("location", {}).get("longitude"),
                    },
                }

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        time.sleep(0.3)

    return list(bars.values())[:max_results]


def get_bar_details(place_id: str) -> dict:
    """
    Get detailed information about a specific bar by its Google Place ID.

    Returns: name, address, phone, website, opening_hours, rating,
    price_level, editorial summary, and a sample of recent reviews.
    """
    field_mask = ",".join([
        "id",
        "displayName",
        "formattedAddress",
        "nationalPhoneNumber",
        "websiteUri",
        "regularOpeningHours",
        "rating",
        "userRatingCount",
        "priceLevel",
        "editorialSummary",
        "reviews",
    ])

    response = requests.get(
        f"{_BASE_URL}/places/{place_id}",
        headers=_headers(field_mask),
    )
    response.raise_for_status()
    place = response.json()

    return {
        "name": place.get("displayName", {}).get("text"),
        "address": place.get("formattedAddress"),
        "phone": place.get("nationalPhoneNumber"),
        "website": place.get("websiteUri"),
        "opening_hours": place.get("regularOpeningHours", {}).get("weekdayDescriptions", []),
        "open_now": place.get("regularOpeningHours", {}).get("openNow"),
        "rating": place.get("rating"),
        "user_ratings_total": place.get("userRatingCount"),
        "price_level": place.get("priceLevel"),
        "summary": place.get("editorialSummary", {}).get("text"),
        "reviews": [
            {"text": r.get("text", {}).get("text"), "rating": r.get("rating")}
            for r in place.get("reviews", [])[:3]
        ],
    }
