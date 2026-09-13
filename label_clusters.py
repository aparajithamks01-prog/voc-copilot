"""
label_clusters.py
-------------------
Adds a short, human-readable label to each cluster based on simple
keyword rules applied to that cluster's summary text.

WHY THIS MATTERS (PM lens):
"Theme 0", "Theme 1", etc. mean nothing to a stakeholder scanning a
chart or table. A short label like "Account Suspensions" or
"Low-Detail Praise" lets someone understand a cluster's content
without reading the full summary -- this is the difference between a
chart stakeholders skim past and one they actually act on.

These rules are intentionally simple (keyword matching, not another
LLM call) because labeling is a low-stakes, high-volume task -- it
doesn't need AI judgment, and keeping it rule-based means it's free,
instant, and fully predictable/debuggable, unlike an LLM call that
could itself need a fact-check.

The rules below were written by reading the actual summaries in this
project's cluster_summaries.csv -- adjust or extend them if you rerun
the pipeline on a different app where the themes differ.

USAGE:
    python label_clusters.py
    (reads cluster_summaries.csv, writes cluster_summaries_labeled.csv)
"""

import pandas as pd

INPUT_FILE = "cluster_summaries.csv"
OUTPUT_FILE = "cluster_summaries_labeled.csv"


def extract_label(summary: str) -> str:
    """
    Returns a short (2-4 word) label describing a cluster's theme,
    based on keyword rules over the cluster's summary text.

    Rules are checked in order -- more specific patterns are checked
    before more generic ones, since a summary can contain multiple
    keywords (e.g. a summary can mention both "suspend" and "bug";
    account access issues should win because they're the more
    actionable/severe theme).
    """
    if not isinstance(summary, str) or not summary.strip():
        return "Uncategorized"

    text = summary.lower()

    # --- Account access & moderation issues (most specific / most actionable) ---
    if any(kw in text for kw in ["suspen", "ban", "disabled profile", "blocked access", "login fail", "lost access"]):
        return "Account Suspensions"

    # --- Technical bugs / app crashes ---
    if any(kw in text for kw in ["crash", "bug", "technical issue", "hang", "freeze", "glitch"]):
        return "Technical Bugs"

    # --- Too vague to summarize (very short/low-signal reviews) ---
    if "too short" in text or "too vague" in text or "vague to summarize" in text:
        return "Low-Detail Praise"

    # --- Distracting / overuse concerns ---
    if "distract" in text or "addictive" in text or "too much time" in text:
        return "Distraction Concerns"

    # --- Educational / social value ---
    if "educat" in text or "social connection" in text or "learn" in text:
        return "Educational & Social Value"

    # --- Mixed sentiment (praise + complaints together) ---
    if ("praise" in text or "positive" in text or "affection" in text) and (
        "complain" in text or "frustrat" in text or "issue" in text or "concern" in text
    ):
        return "Mixed Sentiment"

    # --- General positive sentiment (catch-all for praise-heavy summaries) ---
    if any(kw in text for kw in ["positive sentiment", "praise", "satisfaction", "enthusia", "love it", "best app"]):
        return "General Positive Feedback"

    # --- General negative sentiment (catch-all for complaint-heavy summaries) ---
    if any(kw in text for kw in ["frustrat", "dissatisf", "complain", "negative"]):
        return "General Complaints"

    return "Uncategorized"


def main():
    df = pd.read_csv(INPUT_FILE)

    if "summary" not in df.columns:
        raise ValueError(f"Expected a 'summary' column in {INPUT_FILE}.")

    df["label"] = df["summary"].apply(extract_label)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved labeled clusters to {OUTPUT_FILE}\n")
    print(df[["cluster", "num_reviews", "label"]].to_string(index=False))


if __name__ == "__main__":
    main()