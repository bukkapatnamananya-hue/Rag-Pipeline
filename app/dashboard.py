import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except ImportError:
    px = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluation" / "evaluation_results.json"
QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "evaluation_queries.json"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "evaluation" / "ground_truth.json"

STRATEGY_LABELS = {
    "recursive": "Recursive",
    "structure_aware": "Structure-aware",
    "contextual": "Contextual",
}
STRATEGY_COLORS = {
    "Recursive": "#1E73BE",
    "Structure-aware": "#4A90D9",
    "Contextual": "#123E6A",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def format_number(value, digits=4, suffix="") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def metric_frame(evaluation: dict) -> pd.DataFrame:
    index_data = evaluation.get("index_time", {})
    rows = []
    for strategy, result in evaluation.get("results", {}).items():
        index_stats = index_data.get(strategy, {})
        rows.append(
            {
                "Strategy": STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").title()),
                "Chunks": index_stats.get("chunk_count"),
                "Recall@5": result.get("recall_at_5"),
                "MRR": result.get("mrr"),
                "Mean Query Time (s)": result.get("mean_query_time_seconds"),
                "Mean Embedding Time (s)": result.get("mean_embedding_time_seconds"),
                "Mean Retrieval Time (s)": result.get("mean_retrieval_time_seconds"),
                "Index Time (s)": index_stats.get("index_time_seconds"),
                "_key": strategy,
            }
        )
    columns = [
        "Strategy",
        "Chunks",
        "Recall@5",
        "MRR",
        "Mean Query Time (s)",
        "Mean Embedding Time (s)",
        "Mean Retrieval Time (s)",
        "Index Time (s)",
        "_key",
    ]
    return pd.DataFrame(rows, columns=columns)


def render_chart(data: pd.DataFrame, x: str, y: str, title: str, y_title: str) -> None:
    if data.empty or y not in data or data[y].dropna().empty:
        st.info(f"{title} is not available in the current evaluation output.")
        return

    if px is not None:
        figure = px.bar(
            data,
            x=x,
            y=y,
            color=x,
            color_discrete_map=STRATEGY_COLORS,
            text_auto=".4f",
            title=title,
        )
        figure.update_layout(
            showlegend=False,
            title_font_size=16,
            title_font_color="#0B2A4A",
            font_color="#58718A",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=12, r=12, t=48, b=12),
            yaxis_title=y_title,
            xaxis_title=None,
        )
        figure.update_yaxes(gridcolor="#e8edef", zeroline=False)
        st.plotly_chart(figure, width="stretch")
    else:
        st.caption(title)
        st.bar_chart(data.set_index(x)[y], width="stretch")


def get_retrieved_chunks(query_result: dict) -> list:
    retrieved = query_result.get("retrieved_chunks", [])
    if isinstance(retrieved, list):
        return retrieved[:5]
    return []


def normalize_chunk(item):
    if isinstance(item, dict):
        chunk_id = item.get("chunk_id", item.get("id", "Unknown chunk"))
        score = next(
            (item[key] for key in ("similarity", "similarity_score", "score") if item.get(key) is not None),
            None,
        )
        return str(chunk_id), score
    return str(item), None


def find_answer(query_result: dict):
    for key in ("answer", "generated_answer", "rag_answer"):
        value = query_result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


st.set_page_config(
    page_title="Document RAG Evaluation",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #0B2A4A;
        --deep-blue: #123E6A;
        --blue: #1E73BE;
        --light-blue: #EAF4FF;
        --page-blue: #F5FAFF;
        --text-blue: #17324D;
        --muted-blue: #58718A;
        --line-blue: #D5E7F8;
    }
    .stApp { background: var(--page-blue); color: var(--text-blue); }
    header[data-testid="stHeader"] { background: var(--page-blue); }
    [data-testid="stMainBlockContainer"] {
        max-width: 1500px;
        padding: 4rem clamp(1rem, 3vw, 3rem) 3rem !important;
    }
    h1, h2, h3 { color: var(--navy); letter-spacing: 0; }
    .stApp h1 { font-size: 1.8rem !important; line-height: 1.2; font-weight: 700; }
    .stApp h2 { font-size: 1.35rem !important; font-weight: 680; margin-top: 1.15rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EAF4FF 0%, #F5FAFF 48%, #FFFFFF 100%);
        border-right: 1px solid var(--line-blue);
    }
    [data-testid="stSidebar"] * { color: var(--deep-blue); }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1.25rem .9rem 1rem; }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: .5rem .4rem 1.25rem;
        border-bottom: 1px solid var(--line-blue);
        margin-bottom: 1.2rem;
    }
    .brand-icon {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        flex: 0 0 42px;
        border-radius: 12px;
        background: var(--deep-blue);
        color: white !important;
        font-size: 1.35rem;
    }
    .brand-name { color: var(--navy); font-weight: 800; font-size: .9rem; }
    .brand-subtitle { color: var(--muted-blue); font-size: .78rem; margin-top: 2px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap: 6px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        border-radius: 10px;
        padding: .56rem .7rem;
        margin: 0;
        border: 1px solid transparent;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child { display: none; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p {
        color: var(--deep-blue) !important;
        font-size: .91rem;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: var(--blue);
        border-color: var(--blue);
        box-shadow: 0 5px 12px rgba(30, 115, 190, .18);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) * { color: white !important; }
    .sidebar-footer {
        border-top: 1px solid var(--line-blue);
        margin-top: 1.5rem;
        padding: 1rem .4rem .35rem;
        color: var(--deep-blue);
        font-size: .78rem;
        line-height: 1.55;
    }
    .sidebar-footer strong { display: block; color: var(--navy); font-size: .8rem; }
    .sidebar-version { display: block; color: var(--muted-blue); margin-top: .65rem; }
    .page-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        width: 100%;
        box-sizing: border-box;
        padding: 1.65rem 1.8rem;
        margin: 0 0 1.5rem;
        border: 1px solid #D7EAFE;
        border-radius: 16px 16px 24px 24px;
        background: linear-gradient(112deg, #EAF4FF 0%, #F7FBFF 62%, #FFFFFF 100%);
        box-shadow: 0 8px 22px rgba(11, 42, 74, .055);
    }
    .page-header h1 { margin: .35rem 0 .45rem; color: var(--navy); font-size: 1.8rem; line-height: 1.22; }
    .page-header p { margin: 0; color: var(--muted-blue); font-size: .98rem; line-height: 1.5; }
    .header-kicker { color: var(--blue); font-size: .72rem; font-weight: 800; letter-spacing: .09em; }
    .load-status {
        display: inline-flex;
        align-items: center;
        gap: .55rem;
        flex: 0 0 auto;
        padding: .62rem .85rem;
        border: 1px solid #CBE9DC;
        border-radius: 999px;
        background: #F0FBF5;
        color: #216947;
        font-size: .78rem;
        font-weight: 650;
        white-space: nowrap;
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #35A873; box-shadow: 0 0 0 3px rgba(53, 168, 115, .13); }
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 12px; margin: .25rem 0 1.7rem; }
    .kpi-card {
        min-width: 0;
        min-height: 145px;
        box-sizing: border-box;
        padding: 1rem 1.05rem;
        border: 1px solid var(--line-blue);
        border-radius: 12px;
        background: #FFFFFF;
        box-shadow: 0 5px 15px rgba(18, 62, 106, .055);
    }
    .kpi-icon { color: var(--blue); font-size: 1.2rem; line-height: 1; margin-bottom: .65rem; }
    .kpi-label { color: var(--muted-blue); font-size: .77rem; font-weight: 650; line-height: 1.35; min-height: 2.1em; }
    .kpi-value { color: var(--navy); font-size: 1.65rem; font-weight: 750; line-height: 1.2; overflow-wrap: anywhere; }
    .kpi-description { color: var(--muted-blue); font-size: .71rem; line-height: 1.4; margin-top: .42rem; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line-blue); border-radius: 10px; padding: 16px 18px; box-shadow: 0 5px 15px rgba(18, 62, 106, .045); }
    [data-testid="stMetricLabel"] { color: var(--muted-blue); }
    [data-testid="stMetricValue"] { color: var(--navy); font-size: 1.55rem; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line-blue); border-radius: 9px; }
    [data-testid="stSubheader"] { color: var(--navy); }
    div[data-testid="stAlert"] { border-radius: 9px; }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: 4rem 1rem 2rem !important; }
        .page-header { align-items: flex-start; flex-direction: column; padding: 1.25rem; gap: 1rem; }
        .page-header h1 { font-size: 1.55rem; }
        .load-status { white-space: normal; }
        .kpi-grid { grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="sidebar-brand"><div class="brand-icon">▤</div><div><div class="brand-name">DOCUMENT RAG</div><div class="brand-subtitle">Evaluation Dashboard</div></div></div>',
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Strategy Comparison", "Query Explorer", "Performance"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown(
    '<div class="sidebar-footer"><strong>Zen Technologies FY 2023-24</strong>Retrieval Evaluation<strong style="margin-top:.85rem">RAG Chunking Strategy Evaluation</strong><span class="sidebar-version">v1.0</span></div>',
    unsafe_allow_html=True,
)

missing_files = [path for path in (EVALUATION_PATH, QUERIES_PATH, GROUND_TRUTH_PATH) if not path.is_file()]
if missing_files:
    st.error("Dashboard data file not found. Expected file(s): " + ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing_files))
    st.stop()

try:
    evaluation = load_json(EVALUATION_PATH)
    query_data = load_json(QUERIES_PATH)
    ground_truth = load_json(GROUND_TRUTH_PATH)
except (OSError, json.JSONDecodeError) as error:
    st.error(f"Could not read the evaluation data: {error}")
    st.stop()

metrics = metric_frame(evaluation)
query_items = query_data.get("queries", [])
query_items = [item for item in query_items if isinstance(item, dict) and item.get("id")]
query_items.sort(key=lambda item: int(item["id"][1:]) if item["id"][1:].isdigit() else item["id"])
query_count = len(query_items)

st.markdown(
    '<section class="page-header"><div><div class="header-kicker">RETRIEVAL &amp; PERFORMANCE ANALYSIS</div><h1>Document RAG — Chunking Strategy Evaluation</h1><p>Zen Technologies Annual Report | Retrieval &amp; Performance Analysis</p></div><div class="load-status"><span class="status-dot"></span>Evaluation Results Loaded</div></section>',
    unsafe_allow_html=True,
)

if page == "Overview":
    strategy_count = len(metrics)
    best_recall = metrics["Recall@5"].max() if not metrics.empty else None
    best_mrr = metrics["MRR"].max() if not metrics.empty else None
    lowest_query_time = metrics["Mean Query Time (s)"].min() if not metrics.empty else None
    strategy_names = " · ".join(escape(label) for label in metrics["Strategy"].tolist())
    kpis = [
        ("▦", "Evaluation Queries", str(query_count), "Total queries in evaluation set"),
        ("◈", "Strategies Evaluated", str(strategy_count), strategy_names or "No strategy data available"),
        ("◎", "Highest Recall@5", format_number(best_recall, 2), "Maximum measured retrieval coverage"),
        ("↗", "Highest MRR", format_number(best_mrr, 2), "Maximum measured reciprocal rank"),
        ("◷", "Lowest Mean Query Time", format_number(lowest_query_time, 4) + " s" if lowest_query_time is not None and pd.notna(lowest_query_time) else "N/A", "Fastest measured strategy average"),
    ]
    kpi_markup = "".join(
        f'<article class="kpi-card"><div class="kpi-icon">{icon}</div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-description">{description}</div></article>'
        for icon, label, value, description in kpis
    )
    st.markdown(f'<div class="kpi-grid">{kpi_markup}</div>', unsafe_allow_html=True)

    st.subheader("Strategy Summary")
    overview_columns = ["Strategy", "Chunks", "Recall@5", "MRR", "Mean Query Time (s)", "Index Time (s)"]
    st.dataframe(metrics.reindex(columns=overview_columns), hide_index=True, width="stretch", column_config={
        "Recall@5": st.column_config.NumberColumn(format="%.4f"),
        "MRR": st.column_config.NumberColumn(format="%.4f"),
        "Mean Query Time (s)": st.column_config.NumberColumn(format="%.4f"),
        "Index Time (s)": st.column_config.NumberColumn(format="%.4f"),
    })

    st.subheader("Key Observations")
    if not metrics.empty:
        for _, row in metrics.iterrows():
            if pd.notna(row["Recall@5"]) and row["Recall@5"] == best_recall:
                st.markdown(f"- **{row['Strategy']}** recorded the highest Recall@5 ({row['Recall@5']:.4f}).")
            if pd.notna(row["MRR"]) and row["MRR"] == best_mrr:
                st.markdown(f"- **{row['Strategy']}** recorded the highest MRR ({row['MRR']:.4f}).")
            if pd.notna(row["Mean Query Time (s)"]) and row["Mean Query Time (s)"] == lowest_query_time:
                st.markdown(f"- **{row['Strategy']}** had the lowest mean query time ({row['Mean Query Time (s)']:.4f} s).")
        valid_index = metrics.dropna(subset=["Index Time (s)"])
        valid_chunks = metrics.dropna(subset=["Chunks"])
        if not valid_index.empty:
            fastest_index = valid_index.loc[valid_index["Index Time (s)"].idxmin()]
            st.markdown(f"- **{fastest_index['Strategy']}** had the lowest measured index time ({fastest_index['Index Time (s)']:.4f} s).")
        if len(valid_chunks) > 1:
            fewest = valid_chunks.loc[valid_chunks["Chunks"].idxmin()]
            most = valid_chunks.loc[valid_chunks["Chunks"].idxmax()]
            st.markdown(f"- Chunk counts ranged from **{int(fewest['Chunks'])}** ({fewest['Strategy']}) to **{int(most['Chunks'])}** ({most['Strategy']}).")

elif page == "Strategy Comparison":
    st.subheader("Metric Comparison")
    available_metrics = [
        ("Recall@5", "Recall@5", "Recall@5"),
        ("MRR", "MRR", "MRR"),
        ("Mean Query Time (s)", "Mean Query Time (s)", "Mean query time (seconds)"),
        ("Index Time (s)", "Index Time (s)", "Index time (seconds)"),
    ]
    chosen = st.multiselect(
        "Metrics to display",
        [item[0] for item in available_metrics],
        default=[item[0] for item in available_metrics],
    )
    for selected_metric in chosen:
        metric = next(item for item in available_metrics if item[0] == selected_metric)
        render_chart(metrics, "Strategy", metric[1], metric[2], metric[2])

    st.subheader("Embedding vs Retrieval Time")
    timing_columns = ["Strategy", "Mean Embedding Time (s)", "Mean Retrieval Time (s)"]
    timing_data = metrics.melt(id_vars="Strategy", value_vars=timing_columns[1:], var_name="Timing", value_name="Seconds")
    if not timing_data["Seconds"].dropna().empty:
        if px is not None:
            figure = px.bar(
                timing_data,
                x="Strategy",
                y="Seconds",
                color="Timing",
                barmode="group",
                color_discrete_sequence=["#1E73BE", "#8ABBEA"],
            )
            figure.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#58718A", title_font_color="#0B2A4A", yaxis_title="Mean time (seconds)", xaxis_title=None, legend_title=None)
            figure.update_yaxes(gridcolor="#e8edef", zeroline=False)
            st.plotly_chart(figure, width="stretch")
        else:
            st.bar_chart(timing_data.pivot(index="Strategy", columns="Timing", values="Seconds"), width="stretch")
    else:
        st.info("Embedding and retrieval timing details are not available.")

    st.subheader("Detailed Comparison")
    st.dataframe(metrics.drop(columns="_key", errors="ignore"), hide_index=True, width="stretch")

elif page == "Query Explorer":
    st.subheader("Inspect a query")
    if not query_items:
        st.info("No queries are available in evaluation_queries.json.")
    else:
        selected_id = st.selectbox("Query ID", [item["id"] for item in query_items])
        query_item = next(item for item in query_items if item["id"] == selected_id)
        query_text = query_item.get("query", "Query text not available.")
        st.markdown(f"**{selected_id}**  \n{query_text}")

        strategy_tabs = st.tabs([STRATEGY_LABELS.get(strategy, strategy.title()) for strategy in evaluation.get("strategies", evaluation.get("results", {}).keys())])
        strategy_keys = list(evaluation.get("strategies", evaluation.get("results", {}).keys()))
        answers = {}
        for tab, strategy in zip(strategy_tabs, strategy_keys):
            with tab:
                strategy_result = evaluation.get("results", {}).get(strategy, {})
                query_result = next((item for item in strategy_result.get("queries", []) if item.get("query_id") == selected_id), {})
                relevant_ids = set(ground_truth.get(strategy, {}).get(selected_id, []))
                retrieved = get_retrieved_chunks(query_result)
                summary = st.columns(3)
                summary[0].metric("Recall@5", format_number(query_result.get("recall_at_5")))
                summary[1].metric("MRR", format_number(query_result.get("mrr")))
                summary[2].metric("Query Time", format_number(query_result.get("query_time_seconds"), 4, " s"))

                rows = []
                for rank, item in enumerate(retrieved, start=1):
                    chunk_id, score = normalize_chunk(item)
                    row = {
                        "Rank": rank,
                        "Chunk ID": chunk_id,
                        "Relevance": "Relevant" if chunk_id in relevant_ids else "Not Relevant",
                    }
                    if score is not None:
                        row["Similarity Score"] = score
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows, columns=(list(rows[0]) if rows else ["Rank", "Chunk ID", "Relevance"])), hide_index=True, width="stretch")
                answers[strategy] = find_answer(query_result)

        if any(answers.values()):
            st.subheader("RAG Answer Comparison")
            answer_tabs = st.tabs([STRATEGY_LABELS.get(strategy, strategy.title()) for strategy, answer in answers.items() if answer])
            for tab, (strategy, answer) in zip(answer_tabs, ((key, value) for key, value in answers.items() if value)):
                with tab:
                    st.write(answer)
        else:
            st.caption("RAG answer comparison is not included in the current retrieval evaluation output.")

elif page == "Performance":
    st.subheader("Query Performance")
    query_cards = st.columns(3)
    query_cards[0].metric("Mean Embedding Time", format_number(metrics["Mean Embedding Time (s)"].mean(), 4, " s") if not metrics.empty else "N/A")
    query_cards[1].metric("Mean Retrieval Time", format_number(metrics["Mean Retrieval Time (s)"].mean(), 4, " s") if not metrics.empty else "N/A")
    query_cards[2].metric("Mean Total Query Time", format_number(metrics["Mean Query Time (s)"].mean(), 4, " s") if not metrics.empty else "N/A")
    render_chart(metrics, "Strategy", "Mean Query Time (s)", "Mean Query Time by Strategy", "Seconds")

    st.subheader("Indexing Performance")
    index_cards = st.columns(2)
    index_cards[0].metric("Total Chunks Across Strategies", f"{int(metrics['Chunks'].sum()):,}" if not metrics.empty and metrics["Chunks"].notna().any() else "N/A")
    index_cards[1].metric("Total Index Time", format_number(metrics["Index Time (s)"].sum(), 4, " s") if not metrics.empty and metrics["Index Time (s)"].notna().any() else "N/A")
    render_chart(metrics, "Strategy", "Index Time (s)", "Index Time by Strategy", "Seconds")
    st.dataframe(metrics[["Strategy", "Chunks", "Index Time (s)"]], hide_index=True, width="stretch")
    st.caption("Query time represents the measured embedding + retrieval time used during evaluation. Index time represents the time required to generate embeddings for all chunks for a strategy.")