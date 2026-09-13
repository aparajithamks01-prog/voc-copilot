"""
cluster_reviews.py
--------------------
Step 2 of the VoC Copilot pipeline: turn raw review text into embeddings
and group similar reviews together with KMeans, so the next stage
(LLM summarization) can summarize theme-by-theme instead of reading
500 reviews as one undifferentiated blob.

WHY THIS MATTERS (PM lens):
Clustering is what turns "500 people said stuff" into "these are the
6-8 things people actually said." The cluster count (k) is a judgment
call, not a fact — too few clusters and themes get muddled together,
too many and you're splitting one theme into near-duplicates. Treat
8 as a starting guess to sanity-check against the actual output, not
a fixed answer.

SETUP (run once in your terminal):
    pip install pandas sentence-transformers scikit-learn

USAGE:
    python cluster_reviews.py
    (edit INPUT_FILE, NUM_CLUSTERS below if needed)
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
INPUT_FILE = "reviews.csv"          # CSV produced by the scraping step
TEXT_COLUMN = "content"             # column holding the review text
OUTPUT_FILE = "reviews_clustered.csv"
NUM_CLUSTERS = 8                    # starting guess — see note above
RANDOM_STATE = 42                   # fixed seed so results are reproducible


def load_reviews(path: str, text_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows with missing/empty review text — an embedding model
    # can't do anything useful with a blank string, and letting them
    # through just adds a meaningless cluster member.
    before = len(df)
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with empty '{text_column}' values.")

    return df.reset_index(drop=True)


def embed_reviews(texts, model_name: str = "all-MiniLM-L6-v2"):
    """
    Convert each review into a dense vector that captures its meaning,
    so reviews with similar complaints/praise end up numerically close
    together even if they use different words.
    """
    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(texts)} reviews...")
    embeddings = model.encode(
        list(texts),
        show_progress_bar=True,
        batch_size=32,
    )
    return embeddings


def cluster_embeddings(embeddings, k: int, random_state: int):
    print(f"Running KMeans with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    print("Clustering done.")
    return labels


if __name__ == "__main__":
    df = load_reviews(INPUT_FILE, TEXT_COLUMN)

    embeddings = embed_reviews(df[TEXT_COLUMN])
    df["cluster"] = cluster_embeddings(embeddings, NUM_CLUSTERS, RANDOM_STATE)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(df)} clustered reviews to {OUTPUT_FILE}")
    print("\nCluster sizes:")
    print(df["cluster"].value_counts().sort_index())