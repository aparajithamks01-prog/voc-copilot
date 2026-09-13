"""
fetch_instagram_reviews.py
---------------------------------
Purpose (PM lens):
    This is the first stage of the VoC Copilot pipeline — pulling raw
    customer voice (Play Store reviews) before it gets cleaned,
    embedded/clustered, and summarized downstream.

What it does:
    1. Connects to the Google Play Store via the unofficial
       `google-play-scraper` library (no API key needed).
    2. Fetches ~500 reviews for the Instagram app.
    3. Saves them to reviews.csv with the fields most useful for
       downstream NLP work (review text, rating, date, etc.).

Install dependency (already done in this environment):
    pip install google-play-scraper
"""

import csv
from google_play_scraper import Sort, reviews

# Instagram's Play Store package name (the unique app ID Google uses).
APP_ID = "com.instagram.android"

# How many reviews we want in total.
TARGET_COUNT = 500

# google-play-scraper paginates internally, so we loop and collect
# batches until we hit our target (or the API runs out of reviews).
all_reviews = []
continuation_token = None

while len(all_reviews) < TARGET_COUNT:
    batch, continuation_token = reviews(
        APP_ID,
        lang="en",          # review language
        country="us",       # storefront/country
        sort=Sort.NEWEST,   # most recent reviews first
        count=200,          # reviews per request (max ~200 per call)
        continuation_token=continuation_token,
    )

    if not batch:
        # No more reviews available; stop looping.
        break

    all_reviews.extend(batch)

# Trim to exactly TARGET_COUNT in case the last batch overshot.
all_reviews = all_reviews[:TARGET_COUNT]

# Write the fields most relevant for VoC analysis to CSV.
fieldnames = [
    "reviewId",
    "userName",
    "score",
    "at",
    "content",
    "thumbsUpCount",
    "appVersion",
]

with open("reviews.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in all_reviews:
        writer.writerow(r)

print(f"Saved {len(all_reviews)} reviews to reviews.csv")