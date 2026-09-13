"""
app.py
------
VoC Copilot dashboard: the stakeholder-facing view of the clustered,
summarized, and fact-checked Play Store review data.

WHY THIS MATTERS (PM lens):
Everything upstream (scraping, clustering, summarizing, verifying) is
only useful if a non-technical stakeholder can look at this screen and
immediately understand two things: (1) what customers are saying, and
(2) how much they should trust each summary. Every design choice below
serves one of those two goals -- nothing here is decorative.

SETUP (run once in your terminal):
    pip install streamlit pandas plotly

USAGE:
    streamlit run app.py
    (make sure cluster_summaries.csv is in the same folder)
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from label_clusters import extract_label

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
INPUT_FILE = "cluster_summaries.csv"

COLOR_VERIFIED = "#2E7D32"     # green
COLOR_NEEDS_REVIEW = "#E65100" # amber
COLOR_UNKNOWN = "#9E9E9E"      # grey

st.set_page_config(
    page_title="VoC Copilot — Review Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1150px; }
        [data-testid="stMetric"] {
            background-color: #F7F8FA;
            border: 1px solid #E7E9EE;
            border-radius: 10px;
            padding: 14px 16px 6px 16px;
        }
        [data-testid="stMetricValue"] { font-size: 1.75rem; }
        [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #666; }
        .badge {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-left: 8px;
        }
        .badge-verified { background-color: #E6F4EA; color: #1E4620; }
        .badge-review { background-color: #FDE9D9; color: #7A3E00; }
        .badge-unknown { background-color: #EFEFEF; color: #444; }
        .stExpander { border: 1px solid #E7E9EE !important; border-radius: 10px !important; }
        h1 { font-size: 2rem; margin-bottom: 0.1rem; }
        .subtitle { color: #666; font-size: 1.02rem; margin-bottom: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def classify_status(result: str) -> str:
    """Returns 'verified', 'needs_review', or 'unknown'."""
    if not isinstance(result, str):
        return "unknown"
    text = result.strip().upper()
    if text.startswith("YES") or text.startswith("**YES**"):
        return "verified"
    if text.startswith("NO") or text.startswith("**NO**"):
        return "needs_review"
    return "unknown"


def badge_html(status: str) -> str:
    if status == "verified":
        return '<span class="badge badge-verified">✅ Verified</span>'
    if status == "needs_review":
        return '<span class="badge badge-review">⚠️ Needs Review</span>'
    return '<span class="badge badge-unknown">❓ Unclear</span>'


def extract_unsupported_notes(result: str) -> str:
    """
    The raw verification text mixes a YES/NO verdict with supporting
    detail. For a stakeholder, only the detail matters (the verdict is
    already shown as a badge) -- so strip a leading YES/NO/markdown
    bold marker and return the rest as plain explanatory text.
    """
    if not isinstance(result, str):
        return "No verification notes available."
    text = result.strip()
    for prefix in ["**YES**", "**NO**", "YES", "NO"]:
        if text.upper().startswith(prefix):
            text = text[len(prefix):].strip(" \n.:-")
            break
    return text if text else "No additional notes -- fully consistent with source reviews."


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
try:
    df = load_data(INPUT_FILE)
except FileNotFoundError:
    st.error(
        f"Couldn't find '{INPUT_FILE}'. Make sure it's in the same "
        "folder as this app, then refresh the page."
    )
    st.stop()

df["status"] = df["verification_result"].apply(classify_status)
df["label"] = df["summary"].apply(extract_label)
df["theme_label"] = df.apply(lambda r: f"{r['label']} (Theme {r['cluster']})", axis=1)

STATUS_COLOR = {
    "verified": COLOR_VERIFIED,
    "needs_review": COLOR_NEEDS_REVIEW,
    "unknown": COLOR_UNKNOWN,
}

total_reviews = int(df["num_reviews"].sum())
total_clusters = len(df)
verified_count = int((df["status"] == "verified").sum())
needs_review_count = int((df["status"] == "needs_review").sum())
pass_rate = round(100 * verified_count / total_clusters) if total_clusters else 0

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About this project")
    st.caption(
        "Play Store reviews for Instagram were scraped, grouped into "
        "themes using embeddings and clustering, summarized with an "
        "LLM, and each summary was independently fact-checked against "
        "its own source reviews to catch unsupported claims."
    )

    st.divider()
    st.markdown("### Filters")
    show_only_flagged = st.checkbox("Show only clusters that need review", value=False)
    label_options = sorted(df["label"].unique().tolist())
    selected_labels = st.multiselect(
        "Filter by theme",
        options=label_options,
        default=label_options,
    )
    min_size = st.slider(
        "Minimum reviews in cluster",
        min_value=0,
        max_value=int(df["num_reviews"].max()),
        value=0,
    )

visible_df = df[df["num_reviews"] >= min_size]
visible_df = visible_df[visible_df["label"].isin(selected_labels)]
if show_only_flagged:
    visible_df = visible_df[visible_df["status"] == "needs_review"]
visible_df = visible_df.reset_index(drop=True)

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.title("VoC Copilot")
st.markdown(
    '<div class="subtitle">AI-clustered Instagram Play Store reviews, '
    "summarized and fact-checked for accuracy.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# KEY METRICS
# ---------------------------------------------------------------------
st.header("At a glance")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total reviews analyzed", f"{total_reviews:,}")
m2.metric("Clusters identified", total_clusters)
m3.metric("Summaries verified", f"{pass_rate}%")
m4.metric("Clusters needing review", needs_review_count)

st.write("")

# ---------------------------------------------------------------------
# CHART
# ---------------------------------------------------------------------
st.header("Review volume by cluster")
st.caption("Bar color shows whether that cluster's summary passed fact-checking.")

chart_df = df.sort_values("num_reviews", ascending=True)

fig = go.Figure(
    go.Bar(
        x=chart_df["num_reviews"],
        y=chart_df["theme_label"],
        orientation="h",
        marker_color=[STATUS_COLOR[s] for s in chart_df["status"]],
        text=chart_df["num_reviews"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} reviews<extra></extra>",
    )
)
fig.update_layout(
    height=max(320, 42 * len(chart_df)),
    margin=dict(l=10, r=40, t=10, b=10),
    xaxis_title="Number of reviews",
    yaxis_title=None,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=13),
    showlegend=False,
)
fig.update_xaxes(showgrid=True, gridcolor="#EEE")

st.plotly_chart(fig, use_container_width=True)
st.caption("🟢 Verified &nbsp;&nbsp; 🟠 Needs review &nbsp;&nbsp; ⚪ Unclear", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------
# CLUSTER CARDS
# ---------------------------------------------------------------------
st.header("Cluster details")

if visible_df.empty:
    st.info("No clusters match the current filters. Adjust them in the sidebar.")
else:
    for _, row in visible_df.iterrows():
        header = f"{row['label']}  ·  Theme {row['cluster']}  ·  {int(row['num_reviews'])} reviews"
        with st.expander(header, expanded=False):
            st.markdown(
                f"**Verification status:** {badge_html(row['status'])}",
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown("**Summary**")
            st.write(row["summary"])
            st.write("")
            st.markdown("**Fact-check notes**")
            st.write(extract_unsupported_notes(row["verification_result"]))

st.divider()
st.caption(
    "Verification status reflects whether each AI-generated summary's claims "
    "are directly supported by the sample of reviews it was built from."
)