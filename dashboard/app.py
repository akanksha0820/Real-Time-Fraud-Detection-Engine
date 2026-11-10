"""Polished fraud operations and model-risk command center."""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file with dashboard/ as sys.path[0]. Add the repository
# root so `streamlit run dashboard/app.py` works from a clean shell.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from python.common import ROOT  # noqa: E402
from python.monitoring.operations import OperationsRepository  # noqa: E402

COLORS = {
    "ALLOW": "#22C55E",
    "REVIEW": "#D9A441",
    "BLOCK": "#BF092F",
    "accent": "#3B9797",
    "blue": "#16476A",
    "emerald": "#3B9797",
    "muted": "#9FB0C2",
    "panel": "#172B43",
}


def style_chart(figure, *, height=380):
    """Apply the dashboard's navy / teal visual language to Plotly figures."""
    figure.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=54, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C7D2DF", family="Inter, Segoe UI, sans-serif", size=12),
        title_font=dict(color="#F6F8FB", size=15),
        legend_title_text="",
        hoverlabel=dict(
            bgcolor="#132440",
            bordercolor="#3B9797",
            font_color="#F6F8FB",
        ),
        hovermode="closest",
    )
    figure.update_xaxes(
        gridcolor="rgba(159,176,194,.10)",
        linecolor="rgba(159,176,194,.16)",
        tickfont=dict(color="#9FB0C2"),
        title_font=dict(color="#9FB0C2"),
        zeroline=False,
    )
    figure.update_yaxes(
        gridcolor="rgba(159,176,194,.10)",
        linecolor="rgba(159,176,194,.16)",
        tickfont=dict(color="#9FB0C2"),
        title_font=dict(color="#9FB0C2"),
        zeroline=False,
    )
    return figure


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def money(value) -> str:
    return f"₹{float(value or 0):,.0f}"


def section(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-title">{title}</div>
            <div class="section-copy">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_policy_strip() -> None:
    st.markdown(
        """
        <div class="policy-strip">
            <div class="policy-card allow-card">
                <div class="policy-top"><span class="policy-dot allow-dot"></span>ALLOW</div>
                <div class="policy-copy">Low Operational Risk</div>
            </div>
            <div class="policy-card review-card">
                <div class="policy-top"><span class="policy-dot review-dot"></span>REVIEW</div>
                <div class="policy-copy">Analyst Investigation</div>
            </div>
            <div class="policy-card block-card">
                <div class="policy-top"><span class="policy-dot block-dot"></span>BLOCK</div>
                <div class="policy-copy">High Risk + Supporting Evidence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=120)
def load_offline_data() -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "data/processed/transaction_features.parquet")
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True)
    return frame


st.set_page_config(
    page_title="Fraud Risk Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """

<style>
    :root {
        --crimson:#BF092F;
        --navy:#132440;
        --blue:#16476A;
        --teal:#3B9797;

        --canvas:#0E1A2A;
        --canvas-2:#102137;
        --panel:#172B43;
        --panel-2:#1A314B;
        --panel-soft:#203A55;

        --text:#F4F7FA;
        --text-2:#D6DEE7;
        --muted:#9FB0C2;
        --muted-2:#71869B;

        --border:rgba(192,207,222,.11);
        --border-strong:rgba(59,151,151,.30);
        --shadow:0 14px 34px rgba(3,10,20,.24);
    }

    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", sans-serif;
    }

    .stApp {
        color:var(--text);
        background:
            radial-gradient(circle at 88% -8%, rgba(59,151,151,.11), transparent 27rem),
            radial-gradient(circle at -8% 26%, rgba(22,71,106,.28), transparent 30rem),
            linear-gradient(145deg, #0D1827 0%, #102036 52%, #0D1A2B 100%);
    }

    .stApp::before {
        content:"";
        position:fixed;
        inset:0;
        pointer-events:none !important;
        background-image:
            linear-gradient(rgba(255,255,255,.012) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.012) 1px, transparent 1px);
        background-size:46px 46px;
        mask-image:linear-gradient(to bottom, rgba(0,0,0,.24), transparent 74%);
        z-index:0;
    }

    .block-container {
        position:relative;
        z-index:1;
        max-width:1540px;
        padding-top:3.55rem;
        padding-bottom:3.2rem;
        padding-left:2rem;
        padding-right:2rem;
    }

    /* Keep Streamlit's native sidebar controls usable. */
    #MainMenu, footer, .stDeployButton {
        display:none !important;
    }

    /* Keep Streamlit's header alive because the collapsed-sidebar control
       is mounted there in newer Streamlit versions. We hide individual chrome,
       not the entire toolbar container. */
    header[data-testid="stHeader"] {
        background:transparent !important;
        height:3.1rem !important;
        pointer-events:auto !important;
        z-index:999998 !important;
    }

    [data-testid="stToolbar"] {
        visibility:visible !important;
        height:3.1rem !important;
        background:transparent !important;
        pointer-events:auto !important;
        z-index:999998 !important;
    }

    /* Hide non-essential header actions while preserving sidebar controls. */
    [data-testid="stToolbarActions"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    [data-testid="stDeployButton"],
    .stAppDeployButton,
    button[title="View app source"],
    button[title="Manage app"] {
        display:none !important;
        pointer-events:none !important;
    }

    /* Sidebar collapse button while the sidebar is open. */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebar"] button[kind="header"] {
        visibility:visible !important;
        display:flex !important;
        opacity:1 !important;
    }

    /* Sidebar reopen button while the sidebar is collapsed.
       Multiple selectors cover Streamlit versions. */
    /* Sidebar reopen / collapse controls across Streamlit versions. */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stHeader"] [data-testid="stBaseButton-headerNoPadding"] {
        visibility:visible !important;
        display:flex !important;
        opacity:1 !important;
        pointer-events:auto !important;
        z-index:2147483647 !important;
    }

    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        position:fixed !important;
        top:.72rem !important;
        left:.72rem !important;
        width:40px !important;
        height:40px !important;
        pointer-events:auto !important;
        isolation:isolate !important;
        transform:translateZ(0);
    }

    [data-testid="collapsedControl"] *,
    [data-testid="stSidebarCollapsedControl"] * {
        pointer-events:auto !important;
    }

    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stHeader"] [data-testid="stBaseButton-headerNoPadding"] {
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
        width:40px !important;
        height:40px !important;
        min-width:40px !important;
        min-height:40px !important;
        padding:0 !important;
        background:#132440 !important;
        color:#F4F7FA !important;
        border:1px solid rgba(59,151,151,.42) !important;
        border-radius:11px !important;
        box-shadow:0 10px 24px rgba(0,0,0,.28) !important;
        pointer-events:auto !important;
    }

    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stHeader"] [data-testid="stBaseButton-headerNoPadding"] svg,
    [data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"] svg {
        color:#F4F7FA !important;
        fill:#F4F7FA !important;
        stroke:#F4F7FA !important;
        width:20px !important;
        height:20px !important;
    }

    [data-testid="collapsedControl"]:hover button,
    [data-testid="stSidebarCollapsedControl"]:hover button,
    [data-testid="stHeader"] [data-testid="stBaseButton-headerNoPadding"]:hover {
        background:#16476A !important;
        border-color:#3B9797 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 18% 0%, rgba(59,151,151,.10), transparent 18rem),
            linear-gradient(180deg, #101F34 0%, #0C1929 100%);
        border-right:1px solid rgba(192,207,222,.08);
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top:.7rem;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color:var(--text-2);
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color:#C7D2DF !important;
        font-weight:680 !important;
    }

    .side-brand {
        display:flex;
        align-items:center;
        gap:.75rem;
        padding:.5rem .2rem 1rem;
        border-bottom:1px solid rgba(192,207,222,.08);
        margin-bottom:.8rem;
    }

    .side-mark {
        width:38px;
        height:38px;
        border-radius:12px;
        display:flex;
        align-items:center;
        justify-content:center;
        background:linear-gradient(145deg, var(--blue), var(--teal));
        border:1px solid rgba(255,255,255,.12);
        color:#fff;
        font-size:.72rem;
        font-weight:900;
        letter-spacing:.04em;
        box-shadow:
            0 10px 22px rgba(0,0,0,.22),
            inset 0 1px 0 rgba(255,255,255,.10);
    }

    .side-name {
        color:#F8FAFC;
        font-weight:790;
        font-size:.98rem;
        letter-spacing:-.015em;
    }

    .side-caption {
        color:#8FA3B7;
        font-size:.69rem;
        margin-top:.05rem;
    }

    .side-status {
        margin:.2rem 0 .85rem;
        border:1px solid rgba(192,207,222,.09);
        border-radius:14px;
        padding:.78rem .82rem;
        background:rgba(255,255,255,.035);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.025);
    }

    .side-status-title {
        color:#EEF3F8;
        font-size:.78rem;
        font-weight:730;
    }

    .side-status-copy {
        color:#91A5B9;
        font-size:.70rem;
        margin-top:.18rem;
        line-height:1.48;
    }

    /* Main application header */
    .hero {
        position:relative;
        overflow:hidden;
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
        gap:2rem;
        padding:1.45rem 1.5rem 1.35rem;
        margin:.1rem 0 1.05rem;
        border:1px solid rgba(192,207,222,.10);
        border-radius:19px;
        background:
            radial-gradient(circle at 93% 0%, rgba(59,151,151,.13), transparent 18rem),
            linear-gradient(135deg, rgba(23,43,67,.98), rgba(18,38,61,.98));
        box-shadow:var(--shadow), inset 0 1px 0 rgba(255,255,255,.035);
    }

    .hero::after {
        content:"";
        position:absolute;
        right:-72px;
        top:-105px;
        width:245px;
        height:245px;
        border-radius:50%;
        border:1px solid rgba(59,151,151,.10);
        box-shadow:
            0 0 0 25px rgba(59,151,151,.018),
            0 0 0 50px rgba(22,71,106,.025);
        pointer-events:none;
    }

    .kicker {
        display:inline-flex;
        align-items:center;
        padding:.32rem .58rem;
        border:1px solid rgba(59,151,151,.22);
        border-radius:999px;
        background:rgba(59,151,151,.09);
        color:#7AC1C1;
        font-size:.68rem;
        font-weight:820;
        letter-spacing:.15em;
        text-transform:uppercase;
    }

    .hero h1 {
        color:#F7F9FC;
        font-size:2.42rem;
        font-weight:790;
        margin:.48rem 0 .23rem;
        letter-spacing:-.048em;
        line-height:1.05;
    }

    .subtitle {
        color:#AFC0D0;
        font-size:.94rem;
        max-width:840px;
        line-height:1.58;
    }

    .hero-meta {
        text-align:right;
        color:#8399AE;
        font-size:.69rem;
        line-height:1.62;
        min-width:190px;
        border-left:1px solid rgba(192,207,222,.10);
        padding-left:1rem;
    }

    .hero-meta strong {
        color:#DDE6EE;
        font-weight:680;
    }

    /* Navigation */
    [data-baseweb="tab-list"] {
        gap:.3rem;
        padding:.28rem;
        border:1px solid rgba(192,207,222,.10);
        background:rgba(17,34,55,.86);
        border-radius:13px;
        width:max-content;
        box-shadow:0 8px 22px rgba(0,0,0,.12);
    }

    [data-baseweb="tab"] {
        height:40px;
        border-radius:9px;
        padding-left:1.05rem;
        padding-right:1.05rem;
        color:#91A5B9;
        font-size:.82rem;
        font-weight:690;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        color:#F8FAFC !important;
        background:linear-gradient(135deg, rgba(22,71,106,.95), rgba(59,151,151,.72));
        box-shadow:inset 0 0 0 1px rgba(255,255,255,.045);
    }

    [data-baseweb="tab-highlight"] {
        display:none;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        min-height:108px;
        background:linear-gradient(155deg, rgba(24,46,70,.98), rgba(20,40,63,.98));
        border:1px solid rgba(192,207,222,.10);
        border-top-color:rgba(59,151,151,.34);
        border-radius:15px;
        padding:.95rem 1rem;
        box-shadow:0 10px 26px rgba(0,0,0,.15), inset 0 1px 0 rgba(255,255,255,.025);
        transition:transform .15s ease, border-color .15s ease;
    }

    [data-testid="stMetric"]:hover {
        transform:translateY(-1px);
        border-color:rgba(59,151,151,.24);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color:#9FB0C2 !important;
        font-weight:680 !important;
        font-size:.77rem !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color:#F7F9FC !important;
        letter-spacing:-.035em;
        font-weight:770 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size:.72rem;
    }

    /* Operational status */
    .status-row {
        display:flex;
        gap:.45rem;
        flex-wrap:wrap;
        margin:.72rem 0 .9rem;
    }

    .pill {
        display:inline-flex;
        align-items:center;
        gap:.42rem;
        border:1px solid rgba(192,207,222,.10);
        border-radius:999px;
        padding:.36rem .65rem;
        font-size:.70rem;
        font-weight:720;
        background:rgba(23,43,67,.80);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.02);
    }

    .pill-dot {
        width:7px;
        height:7px;
        border-radius:50%;
        background:currentColor;
        box-shadow:0 0 8px currentColor;
        opacity:.9;
    }

    .ok {
        color:#66B8B8 !important;
        border-color:rgba(59,151,151,.28) !important;
        background:rgba(59,151,151,.08) !important;
    }

    .warn {
        color:#E6B854 !important;
        border-color:rgba(217,164,65,.26) !important;
        background:rgba(217,164,65,.08) !important;
    }

    .bad {
        color:#F05C79 !important;
        border-color:rgba(191,9,47,.30) !important;
        background:rgba(191,9,47,.10) !important;
    }

    /* Decision policy */
    .policy-strip {
        display:grid;
        grid-template-columns:repeat(3,1fr);
        gap:.58rem;
        margin:.22rem 0 1.1rem;
    }

    .policy-card {
        min-height:84px;
        border:1px solid rgba(192,207,222,.10);
        border-radius:14px;
        background:linear-gradient(150deg, rgba(24,46,70,.96), rgba(20,39,61,.96));
        padding:.85rem .9rem;
        box-shadow:0 8px 22px rgba(0,0,0,.12), inset 0 1px 0 rgba(255,255,255,.02);
    }

    .policy-top {
        color:#F4F7FA;
        font-size:.74rem;
        font-weight:830;
        letter-spacing:.055em;
    }

    .policy-copy {
        color:#95A8BB;
        font-size:.70rem;
        margin-top:.22rem;
    }

    .policy-dot {
        display:inline-block;
        width:7px;
        height:7px;
        border-radius:50%;
        margin-right:.42rem;
    }

    .allow-dot {
        background:#22C55E;
        box-shadow:0 0 9px rgba(34,197,94,.55);
    }

    .review-dot {
        background:#D9A441;
        box-shadow:0 0 8px rgba(217,164,65,.48);
    }

    .block-dot {
        background:var(--crimson);
        box-shadow:0 0 8px rgba(191,9,47,.55);
    }

    .allow-card {
        border-left:3px solid #22C55E;
        background:
            linear-gradient(90deg, rgba(34,197,94,.055), transparent 32%),
            linear-gradient(150deg, rgba(24,46,70,.96), rgba(20,39,61,.96));
    }
    .review-card { border-left:3px solid #D9A441; }
    .block-card { border-left:3px solid var(--crimson); }

    /* Sections */
    .section-head {
        margin:1.34rem 0 .62rem;
    }

    .section-title {
        color:#F4F7FA;
        font-size:1.02rem;
        font-weight:750;
        letter-spacing:-.015em;
    }

    .section-copy {
        color:#91A5B9;
        font-size:.76rem;
        margin-top:.13rem;
        line-height:1.48;
    }

    .demo-note {
        color:#7F95AA;
        font-size:.68rem;
        line-height:1.5;
    }

    hr {
        border-color:rgba(192,207,222,.08) !important;
        margin:.8rem 0 !important;
    }

    /* Charts + tables */
    div[data-testid="stPlotlyChart"] {
        background:linear-gradient(150deg, rgba(23,43,67,.97), rgba(18,38,61,.97));
        border:1px solid rgba(192,207,222,.10);
        border-radius:15px;
        box-shadow:0 10px 26px rgba(0,0,0,.14);
        overflow:hidden;
        padding:.15rem;
    }

    div[data-testid="stDataFrame"] {
        border:1px solid rgba(192,207,222,.10);
        border-radius:14px;
        overflow:hidden;
        background:#172B43;
        box-shadow:0 10px 24px rgba(0,0,0,.12);
    }

    /* Keep ordinary Streamlit copy readable. */
    .stMarkdown,
    .stMarkdown p,
    .stCaption,
    [data-testid="stCaptionContainer"],
    [data-testid="stText"] {
        color:var(--text-2);
    }

    h1, h2, h3, h4, h5, h6 {
        color:var(--text) !important;
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background:#172B43 !important;
        color:#F4F7FA !important;
        border-color:rgba(192,207,222,.12) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg {
        color:#D6DEE7 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius:10px;
        min-height:40px;
        font-weight:740;
        letter-spacing:.005em;
        color:#F8FAFC;
        border:1px solid rgba(192,207,222,.12);
        background:#1A314B;
        box-shadow:none;
    }

    .stButton > button:hover {
        border-color:rgba(59,151,151,.34);
        color:#FFFFFF;
    }

    .stButton > button[kind="primary"] {
        background:linear-gradient(135deg, #16476A, #3B9797);
        border:1px solid rgba(59,151,151,.30);
        color:#FFFFFF;
        box-shadow:0 9px 20px rgba(0,0,0,.14) !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background:rgba(23,43,67,.72);
        border:1px solid rgba(192,207,222,.09);
        border-radius:12px;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left:1rem;
            padding-right:1rem;
        }

        .hero {
            display:block;
            padding:1.15rem 1rem;
        }

        .hero h1 {
            font-size:2rem;
        }

        .hero-meta {
            text-align:left;
            border-left:0;
            padding-left:0;
            margin-top:.75rem;
        }

        .policy-strip {
            grid-template-columns:1fr;
        }

        [data-baseweb="tab-list"] {
            width:100%;
            overflow-x:auto;
        }
    }

    [data-testid="stHeader"] > div,
    [data-testid="stToolbar"] > div {
        pointer-events:auto !important;
    }

</style>

""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div>
    <div class="kicker">Real-Time Risk Operations</div>
    <h1>Fraud Risk Command Center</h1>
    <div class="subtitle">Operational view of decision traffic, investigations, pipeline integrity, merchant concentration and model governance.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="side-mark">FR</div>
            <div>
                <div class="side-name">Risk Engine</div>
                <div class="side-caption">Fraud Operations Workspace</div>
            </div>
        </div>
        <div class="side-status">
            <div class="side-status-title"><span class="live-dot"></span>Operations Workspace</div>
            <div class="side-status-copy">Streaming Decisions, Investigations and Model Oversight</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("##### Controls")
    refresh = st.button("Refresh Live Data", type="primary", use_container_width=True)
    st.caption("Live data refreshes on demand. Offline data is cached for two minutes.")
    st.divider()
    st.markdown(
        '<div class="demo-note">Portfolio Demonstration<br>Synthetic Transaction Data Only</div>',
        unsafe_allow_html=True,
    )

if refresh:
    st.cache_data.clear()
    st.rerun()

live, investigations, model_context = st.tabs(
    ["Live Operations", "Investigations", "Simulation & Model"]
)

with live:
    try:
        repository = OperationsRepository()
        summary = repository.summary()
        pipeline = summary["pipeline"]
        decisions = numeric(pd.DataFrame(summary["decisions"]), ["transactions", "amount_inr"])
        queue = numeric(pd.DataFrame(repository.review_queue(200)), ["amount_inr", "fraud_probability", "age_minutes"])
        reconciled = pipeline["persisted_transactions"] == pipeline["persisted_decisions"]
        backlog = int(pipeline["pending_outbox"] or 0)
        dead_letters = int(pipeline["dead_letters"] or 0)

        badges = [
            f'<span class="pill {"ok" if reconciled else "bad"}"><span class="pill-dot"></span>Ledger {"Reconciled" if reconciled else "Mismatch"}</span>',
            f'<span class="pill {"ok" if backlog == 0 else "warn"}"><span class="pill-dot"></span>Outbox {"Clear" if backlog == 0 else f"{backlog} Pending"}</span>',
            f'<span class="pill {"ok" if dead_letters == 0 else "warn"}"><span class="pill-dot"></span>DLQ {"Clear" if dead_letters == 0 else dead_letters}</span>',
        ]
        st.markdown('<div class="status-row">' + "".join(badges) + "</div>", unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Consumed Events", f"{int(pipeline['consumed_messages']):,}")
        c2.metric("Unique Transactions", f"{int(pipeline['persisted_transactions']):,}")
        c3.metric("Risk Decisions", f"{int(pipeline['persisted_decisions']):,}")
        c4.metric("Open Review Cases", f"{len(queue):,}")
        c5.metric("Transaction Value", money(decisions["amount_inr"].sum() if not decisions.empty else 0))

        decision_policy_strip()

        if not reconciled:
            st.error("Transaction/decision reconciliation failed. Pause event processing and investigate.")
        if backlog:
            st.warning("Outbound decisions are waiting for Kafka publication.")
        if dead_letters:
            st.warning(f"{dead_letters:,} dead-letter event(s) require contract review.")

        section("Decision Traffic", "Current durable decisions and the value routed through each operational action.")
        left, right = st.columns([.9, 1.35])
        if decisions.empty:
            left.info("No decisions have been persisted yet.")
            right.info("Produce or score transactions to populate the live view.")
        else:
            decision_order = [x for x in ["ALLOW", "REVIEW", "BLOCK"] if x in decisions["decision"].values]
            donut = px.pie(
                decisions,
                names="decision",
                values="transactions",
                hole=.70,
                color="decision",
                color_discrete_map=COLORS,
                category_orders={"decision": decision_order},
                title="Decision Mix",
            )
            donut.update_traces(
                textposition="outside",
                textinfo="percent+label",
                marker_line_color="#ffffff",
                marker_line_width=2,
                pull=[0.01] * len(decision_order),
            )
            left.plotly_chart(style_chart(donut), use_container_width=True)

            value_chart = px.bar(
                decisions,
                x="decision",
                y="amount_inr",
                color="decision",
                color_discrete_map=COLORS,
                text_auto=".3s",
                category_orders={"decision": decision_order},
                title="Transaction Value by decision",
                labels={"amount_inr": "Value (₹)", "decision": ""},
            )
            value_chart.update_traces(marker_line_width=0, textposition="outside", opacity=.9)
            value_chart.update_layout(showlegend=False, bargap=.46)
            right.plotly_chart(style_chart(value_chart), use_container_width=True)

        try:
            hourly = numeric(pd.DataFrame(repository.hourly_decisions()), ["transactions", "amount_inr", "average_probability"])
            if not hourly.empty:
                hourly["metric_hour"] = pd.to_datetime(hourly["metric_hour"], utc=True)
                hourly = hourly.sort_values("metric_hour")
                trend = px.area(
                    hourly,
                    x="metric_hour",
                    y="transactions",
                    color="decision",
                    color_discrete_map=COLORS,
                    title="Hourly Decision Throughput",
                    labels={"metric_hour": "", "transactions": "Transactions"},
                )
                trend.update_traces(mode="lines", line_width=2)
                st.plotly_chart(style_chart(trend, height=350), use_container_width=True)
        except Exception:
            st.caption("Hourly trend becomes available after the latest SQL view is applied.")

    except Exception as exc:
        st.error("The live operations layer is unavailable. Verify PostgreSQL and POSTGRES_DSN.")
        with st.expander("Technical detail"):
            st.code(f"{type(exc).__name__}: {exc}")

with investigations:
    try:
        repository = OperationsRepository()
        queue = numeric(pd.DataFrame(repository.review_queue(200)), ["amount_inr", "fraud_probability", "age_minutes"])
        merchants = numeric(
            pd.DataFrame(repository.merchant_risk(50)),
            ["transactions", "amount_inr", "blocks", "reviews", "average_probability"],
        )

        section("Analyst Workload", "Prioritize cases by risk, value and time waiting for investigation.")
        if queue.empty:
            st.success("No open or in-progress review cases.", icon="✓")
        else:
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Cases Awaiting Action", f"{len(queue):,}")
            q2.metric("Review Value", money(queue["amount_inr"].sum()))
            q3.metric("Average Risk", f"{queue['fraud_probability'].mean():.1%}")
            q4.metric("Oldest Case", f"{queue['age_minutes'].max():,.0f} min")

            priority = queue.groupby("priority", dropna=False).size().rename("cases").reset_index()
            priority_order = [x for x in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] if x in priority["priority"].values]
            priority_chart = px.bar(
                priority,
                x="priority",
                y="cases",
                color="priority",
                category_orders={"priority": priority_order},
                color_discrete_map={"CRITICAL": "#BF092F", "HIGH": "#f97316", "MEDIUM": "#f59e0b", "LOW": "#3B9797"},
                title="Open Cases by Priority",
                text_auto=True,
            )
            priority_chart.update_layout(showlegend=False, bargap=.46)
            st.plotly_chart(style_chart(priority_chart, height=320), use_container_width=True)

            display_columns = [
                column for column in [
                    "case_id", "priority", "status", "fraud_probability", "amount_inr",
                    "age_minutes", "reason_codes", "assigned_to", "created_at"
                ] if column in queue
            ]
            st.dataframe(
                queue[display_columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "fraud_probability": st.column_config.ProgressColumn("Risk", min_value=0.0, max_value=1.0, format="percent"),
                    "amount_inr": st.column_config.NumberColumn("Amount", format="₹ %.2f"),
                    "age_minutes": st.column_config.NumberColumn("Age (min)", format="%.0f"),
                },
            )

        section("Merchant Concentration", "Find merchants combining elevated probability, transaction value and review volume.")
        if merchants.empty:
            st.info("Merchant risk data will appear after transactions are persisted.")
        else:
            scatter = px.scatter(
                merchants,
                x="amount_inr",
                y="average_probability",
                size="transactions",
                color="reviews",
                hover_name="merchant_id",
                hover_data=["transactions", "blocks", "reviews"],
                color_continuous_scale=["#16476A", "#16476A", "#f59e0b"],
                title="Merchant Exposure Map",
                labels={"amount_inr": "Transaction Value (₹)", "average_probability": "Average Fraud Probability", "reviews": "Reviews"},
            )
            scatter.update_yaxes(tickformat=".1%")
            st.plotly_chart(style_chart(scatter, height=430), use_container_width=True)
            st.dataframe(
                merchants.head(25),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "amount_inr": st.column_config.NumberColumn("Value", format="₹ %.2f"),
                    "average_probability": st.column_config.ProgressColumn("Average Risk", min_value=0.0, max_value=1.0, format="percent"),
                },
            )

    except Exception as exc:
        st.error("Investigation data is unavailable. Verify PostgreSQL and apply the latest schema.")
        with st.expander("Technical detail"):
            st.code(f"{type(exc).__name__}: {exc}")

with model_context:
    try:
        data = load_offline_data()
        fraud = data[data["is_fraud"].eq(1)]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Simulated Events", f"{len(data):,}")
        c2.metric("Fraud Rate", f"{data['is_fraud'].mean():.2%}")
        c3.metric("Fraud Exposure", money(fraud["amount_inr"].sum()))
        c4.metric("Customers", f"{data['customer_id'].nunique():,}")
        c5.metric("High-Risk Merchants", f"{data.loc[data['risk_tier'].eq('high'), 'merchant_id'].nunique():,}")

        section("Temporal Behavior", "Two-year simulation window; rolling fraud rate reduces day-to-day noise.")
        daily = (
            data.set_index("event_timestamp")
            .resample("D")
            .agg(transactions=("transaction_id", "size"), fraud_rate=("is_fraud", "mean"), amount_inr=("amount_inr", "sum"))
            .reset_index()
        )
        daily["fraud_rate_30d"] = daily["fraud_rate"].rolling(30, min_periods=1).mean()

        left, right = st.columns([1.35, .85])
        rate_chart = px.line(
            daily,
            x="event_timestamp",
            y="fraud_rate_30d",
            title="30-Day Rolling Fraud Rate",
            labels={"event_timestamp": "", "fraud_rate_30d": "Fraud Rate"},
        )
        rate_chart.update_traces(line_color=COLORS["accent"], line_width=2.2, fill="tozeroy", fillcolor="rgba(79,70,229,.06)")
        rate_chart.update_yaxes(tickformat=".2%")
        left.plotly_chart(style_chart(rate_chart), use_container_width=True)

        amount_sample = data.sample(min(len(data), 100_000), random_state=42).copy()
        amount_sample["fraud_label"] = amount_sample["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
        volume_chart = px.histogram(
            amount_sample,
            x="amount_inr",
            color="fraud_label",
            nbins=60,
            log_y=True,
            color_discrete_map={"Legitimate": "#22C55E", "Fraud": "#BF092F"},
            title="Amount Distribution (Log Scale)",
            labels={"amount_inr": "Amount (₹)", "count": "Events", "fraud_label": "Class"},
        )
        volume_chart.update_traces(opacity=.82)
        right.plotly_chart(style_chart(volume_chart), use_container_width=True)

        scenario_column = "fraud_scenario"
        if scenario_column in fraud and fraud[scenario_column].notna().any():
            section("Attack Scenarios", "Scenario volume and financial exposure show where the model must capture different attack mechanics.")
            scenarios = (
                fraud.groupby(scenario_column)
                .agg(transactions=("transaction_id", "size"), exposure_inr=("amount_inr", "sum"))
                .reset_index()
                .sort_values("transactions", ascending=True)
            )
            left, right = st.columns(2)
            count_chart = px.bar(
                scenarios,
                x="transactions",
                y=scenario_column,
                orientation="h",
                color="transactions",
                color_continuous_scale=["#16476A", "#3B9797"],
                title="Fraud Events by Scenario",
                labels={scenario_column: "", "transactions": "Fraud events"},
            )
            count_chart.update_layout(coloraxis_showscale=False)
            left.plotly_chart(style_chart(count_chart), use_container_width=True)

            exposure_chart = px.bar(
                scenarios.sort_values("exposure_inr", ascending=True),
                x="exposure_inr",
                y=scenario_column,
                orientation="h",
                color="exposure_inr",
                color_continuous_scale=["#f59e0b", "#BF092F"],
                title="Fraud Exposure by scenario",
                labels={scenario_column: "", "exposure_inr": "Exposure (₹)"},
            )
            exposure_chart.update_layout(coloraxis_showscale=False)
            right.plotly_chart(style_chart(exposure_chart), use_container_width=True)

        if "legitimate_pattern" in data:
            hard = data[data["legitimate_pattern"].ne("routine") & data["legitimate_pattern"].notna()]
            if not hard.empty:
                section("Legitimate Hard Negatives", "Fraud-like legitimate behavior prevents the model from relying on easy shortcuts.")
                patterns = hard["legitimate_pattern"].value_counts().rename_axis("pattern").reset_index(name="transactions")
                hard_chart = px.treemap(
                    patterns,
                    path=["pattern"],
                    values="transactions",
                    color="transactions",
                    color_continuous_scale=["#16476A", "#3B9797"],
                    title="Legitimate Fraud-Lookalike Activity",
                )
                hard_chart.update_layout(coloraxis_showscale=False)
                st.plotly_chart(style_chart(hard_chart, height=410), use_container_width=True)

        section("Model Governance", "Registered versions and delayed-label monitoring evidence from the operational ledger.")
        try:
            governance = OperationsRepository().model_governance()
            registry = pd.DataFrame(governance["registry"])
            monitoring = pd.DataFrame(governance["monitoring_runs"])
            left, right = st.columns(2)
            with left:
                st.markdown("##### Champion / Challenger Registry")
                if registry.empty:
                    st.info("No registered models yet.")
                else:
                    st.dataframe(registry, use_container_width=True, hide_index=True)
            with right:
                st.markdown("##### Delayed-Label Monitoring")
                if monitoring.empty:
                    st.info("No matured-label monitoring runs yet.")
                else:
                    st.dataframe(monitoring, use_container_width=True, hide_index=True)
        except Exception:
            st.info("Operational governance tables are unavailable; offline simulation analysis remains accessible.")

    except FileNotFoundError:
        st.error("Processed feature data is missing. Run `python -m python.features.build_features` first.")
    except Exception as exc:
        st.error("Offline model context could not be loaded.")
        with st.expander("Technical detail"):
            st.code(f"{type(exc).__name__}: {exc}")
