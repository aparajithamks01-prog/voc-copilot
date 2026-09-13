"""
summarize_clusters.py
------------------------
Step 3 of the VoC Copilot pipeline: turn each cluster of similar
reviews into a short, human-readable theme summary using the Gemini API.

WHY THIS MATTERS (PM lens):
This is the step that actually produces something a stakeholder can
read in a standup -- "Cluster 7 = users frustrated with delivery
delays" is far more useful than a spreadsheet of 500 raw reviews.
Sampling 15 reviews per cluster (instead of sending every review) keeps
API cost and latency down while still giving the model enough signal
to describe the theme accurately -- sample size is a cost/accuracy
tradeoff, not a fixed rule.

This version adds a hallucination-detection guardrail: after each
summary is generated, a second Gemini call re-reads the same reviews
and checks whether the summary only makes claims those reviews
actually support. This matters because LLM summaries can quietly
invent or overgeneralize claims that sound plausible but aren't
backed by the source text -- and a PM presenting an unverified
summary to stakeholders has no way to catch that on their own.

SETUP (run once in your terminal):
    pip install google-genai python-dotenv

    Create a file called .env in this same folder containing:
        GEMINI_API_KEY=your_key_here

    Get a free key at: https://aistudio.google.com/apikey

USAGE:
    python summarize_clusters.py
    (edit INPUT_FILE, SAMPLE_SIZE below if needed)
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
INPUT_FILE = "reviews_clustered.csv"   # output of the clustering step
TEXT_COLUMN = "content"
CLUSTER_COLUMN = "cluster"
OUTPUT_FILE = "cluster_summaries.csv"
SAMPLE_SIZE = 15                       # reviews sent to Gemini per cluster
MODEL = "gemini-3.5-flash-lite"        # much higher free-tier daily quota than 3.6-flash
RANDOM_STATE = 42                      # fixed seed so sampling is reproducible
MAX_RETRIES = 5                        # retries for transient server errors (503, etc.)
RETRY_BASE_DELAY = 10                  # seconds -- doubles each retry (10, 20, 40, ...)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print("Key loaded:", bool(api_key), "| length:", len(api_key) if api_key else 0)

client = genai.Client(api_key=api_key)


def call_gemini_with_retry(prompt: str) -> str:
    """
    Wraps a Gemini call with retry + exponential backoff.

    Free-tier and even paid Gemini endpoints occasionally return 503
    (server overloaded) or 429 (rate limited) errors that have nothing
    to do with your code -- they're transient and usually resolve
    within seconds to a couple minutes. Retrying automatically means
    one busy moment on Google's end doesn't kill a run that's already
    processed several clusters.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=200),
            )
            return response.text.strip()

        except errors.ServerError as e:
            last_error = e
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"  Server busy (attempt {attempt}/{MAX_RETRIES}). "
                  f"Retrying in {delay}s...")
            time.sleep(delay)

        except errors.ClientError as e:
            error_str = str(e)

            # A daily quota (RPD) error won't resolve by waiting a few
            # seconds -- it resets at midnight Pacific Time, so retrying
            # here just burns time. Fail fast with a clear message.
            if "PerDay" in error_str or "RequestsPerDay" in error_str:
                raise RuntimeError(
                    "Daily free-tier quota exceeded for this model. "
                    "This resets at midnight Pacific Time -- wait and "
                    "rerun, or switch to a model with a higher daily "
                    "quota. Already-completed clusters were saved to "
                    f"{OUTPUT_FILE}."
                ) from e

            # 429 rate limit (per-minute) is worth retrying; other 4xx
            # errors (bad request, auth, etc.) won't fix themselves.
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                last_error = e
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"  Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                      f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise

    raise last_error


def build_prompt(reviews_sample: list) -> str:
    numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reviews_sample))
    return (
        "Here are sample customer reviews that were grouped together "
        "because they discuss a similar theme:\n\n"
        f"{numbered}\n\n"
        "In 2-3 sentences, summarize the common theme these reviews "
        "share. Focus on what users are saying and why it matters -- "
        "not on the fact that they're reviews. Do not list individual "
        "reviews back; write a synthesized summary.\n\n"
        "Only state what is explicitly said in the reviews. Do not "
        "infer satisfaction levels, quality judgments, or reasons "
        "unless directly stated. Do not generalize a claim mentioned "
        "by only one review as if it applies to most or all reviews "
        "in the group. If the reviews are too short or vague to "
        "summarize meaningfully, say so instead of guessing."
    )


def summarize_cluster(reviews_sample: list) -> str:
    prompt = build_prompt(reviews_sample)
    return call_gemini_with_retry(prompt)


def build_verification_prompt(summary: str, reviews_sample: list) -> str:
    numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reviews_sample))
    return (
        "Here is a summary that was generated from a set of customer "
        "reviews, followed by the original reviews themselves.\n\n"
        f"SUMMARY:\n{summary}\n\n"
        f"ORIGINAL REVIEWS:\n{numbered}\n\n"
        "Does this summary only contain claims that are directly "
        "supported by these reviews? Answer YES or NO, and if NO, "
        "list which claims are unsupported."
    )


def verify_summary(summary: str, reviews_sample: list) -> str:
    """
    Hallucination-detection guardrail: re-checks a generated summary
    against the same reviews it was built from, so ungrounded claims
    get flagged instead of silently shipped to a stakeholder.
    """
    prompt = build_verification_prompt(summary, reviews_sample)
    return call_gemini_with_retry(prompt)


def main():
    df = pd.read_csv(INPUT_FILE)

    if CLUSTER_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Expected columns '{CLUSTER_COLUMN}' and '{TEXT_COLUMN}' "
            f"in {INPUT_FILE}. Found: {list(df.columns)}"
        )

    summaries = []

    for cluster_id, group in df.groupby(CLUSTER_COLUMN):
        # Sample up to SAMPLE_SIZE reviews -- if a cluster has fewer,
        # just use all of them.
        sample_n = min(SAMPLE_SIZE, len(group))
        sample = group[TEXT_COLUMN].dropna().sample(
            n=sample_n, random_state=RANDOM_STATE
        ).tolist()

        print(f"Summarizing cluster {cluster_id} ({len(group)} reviews, "
              f"sampling {sample_n})...")

        summary = summarize_cluster(sample)

        # Pause between the summary call and the verification call --
        # each cluster now makes two API calls instead of one.
        time.sleep(5)

        print(f"Verifying summary for cluster {cluster_id}...")
        verification_result = verify_summary(summary, sample)

        summaries.append({
            "cluster": cluster_id,
            "num_reviews": len(group),
            "sample_size": sample_n,
            "summary": summary,
            "verification_result": verification_result,
        })

        # Save progress after every cluster -- if a later cluster fails
        # even after retries, you don't lose the work already done.
        pd.DataFrame(summaries).to_csv(OUTPUT_FILE, index=False)

        # Small pause to stay comfortably under free-tier rate limits
        # (free tier is roughly 5-15 requests/minute for Flash models)
        time.sleep(5)

    out_df = pd.DataFrame(summaries)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(out_df)} cluster summaries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()