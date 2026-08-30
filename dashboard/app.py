"""
Day 7: Smart Recovery Engine dashboard.

The live demo interface for the Razorpay hackathon. Pulls together
everything built Days 1-6 into one interactive app: headline results,
decline-code insights, a live "what would the engine decide" simulator,
a transaction explorer, and an honest methodology/assumptions section.

Run with:
    streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rule_scheduler import (HARD_OVERRIDE_CODES, NUDGE_INSTEAD_OF_RETRY,  # noqa: E402
                              RETRY_THRESHOLD, SALARY_WINDOW_DAYS, find_best_retry_day)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "generated"
TAXONOMY_PATH = ROOT / "data" / "decline_codes.json"

CARD_CODES = ["INSUFFICIENT_FUNDS", "ISSUER_TIMEOUT", "DO_NOT_HONOR_TEMP",
              "CARD_EXPIRED", "CARD_LOST_STOLEN", "INVALID_CARD_DETAILS",
              "BANK_SERVER_DOWN"]
UPI_CODES = ["UPI_MANDATE_NOT_CONFIRMED", "UPI_PSP_APP_ERROR", "UPI_MANDATE_REVOKED"]

NUDGE_SUCCESS_ASSUMPTION = 0.55
ESCALATION_SUCCESS_ASSUMPTION = 0.25

# ---------------------------------------------------------------------------
# Page config + fintech-style theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Recovery Engine | Razorpay Hackathon",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRIMARY = "#3395FF"      # Razorpay-style blue
DARK_BG = "#0E1117"
CARD_BG = "#1A2233"
BORDER = "#2D3748"
TEXT_LIGHT = "#F1F5F9"
TEXT_MUTED = "#94A3B8"
SUCCESS = "#22C55E"
DANGER = "#EF4444"
WARNING = "#F59E0B"

st.markdown(f"""
<style>
    .stApp {{ background-color: {DARK_BG}; }}
    h1, h2, h3, h4, h5, p, span, label {{ color: {TEXT_LIGHT}; }}
    [data-testid="stMetric"] {{
        background-color: {CARD_BG}; padding: 1rem 1.2rem; border-radius: 10px;
        border: 1px solid {BORDER};
    }}
    [data-testid="stMetricLabel"] p {{ color: {TEXT_MUTED} !important; font-size: 0.9rem; }}
    [data-testid="stMetricValue"] {{ color: {TEXT_LIGHT} !important; font-size: 2rem; }}
    [data-testid="stMetricDelta"] {{ font-size: 0.95rem; }}
    .stTabs [data-baseweb="tab"] {{ font-size: 1rem; font-weight: 600; color: {TEXT_MUTED}; }}
    .stTabs [aria-selected="true"] {{ color: {PRIMARY} !important; }}
    .badge {{
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; margin-right: 6px;
    }}
    .badge-retry {{ background-color: rgba(34,197,94,0.15); color: {SUCCESS}; }}
    .badge-nudge {{ background-color: rgba(51,149,255,0.15); color: {PRIMARY}; }}
    .badge-escalate {{ background-color: rgba(239,68,68,0.15); color: {DANGER}; }}
    .badge-exclude {{ background-color: rgba(148,163,184,0.15); color: {TEXT_MUTED}; }}
    .assumption-box {{
        background-color: rgba(245,158,11,0.1); border-left: 4px solid {WARNING};
        padding: 12px 16px; border-radius: 6px; margin: 10px 0;
    }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached so the app stays fast)
# ---------------------------------------------------------------------------
@st.cache_data
def load_taxonomy():
    with open(TAXONOMY_PATH, "r") as f:
        return json.load(f)["decline_codes"]


@st.cache_data
def load_transactions():
    df = pd.read_csv(DATA_DIR / "transactions.csv")
    for col in ["is_voluntary_churn", "is_salary_window", "retry_success"]:
        df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
    return df


@st.cache_data
def load_scheduled_actions():
    return pd.read_csv(DATA_DIR / "scheduled_actions.csv")


@st.cache_data
def load_simulation_results():
    return pd.read_csv(DATA_DIR / "simulation_results.csv")


@st.cache_resource
def load_model_bundle():
    return joblib.load(DATA_DIR / "model.pkl")


def missing_data_warning(missing_file):
    st.error(
        f"**Missing `{missing_file}`.** Run the pipeline scripts in order first: "
        f"`generate_data.py` -> `build_features.py` -> `train_model.py` -> "
        f"`rule_scheduler.py` -> `simulate_recovery.py`."
    )
    st.stop()


def apply_dark_theme(fig, **kwargs):
    """Applies consistent dark styling to a Plotly figure so charts don't
    render as bright white boxes against the dashboard's dark background."""
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT_LIGHT, title_font_color=TEXT_LIGHT,
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **kwargs,
    )
    return fig


try:
    taxonomy = load_taxonomy()
    taxonomy_by_code = {c["code"]: c for c in taxonomy}
    transactions = load_transactions()
    scheduled = load_scheduled_actions()
    sim_results = load_simulation_results()
    model_bundle = load_model_bundle()
except FileNotFoundError as e:
    missing_data_warning(str(e))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Smart Recovery Engine")
st.markdown(
    "##### Decline-aware payment recovery - stop retrying every failure the same way"
)
st.write("")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_insights, tab_simulator, tab_explorer, tab_methodology = st.tabs(
    ["📊 Overview", "🔍 Decline Insights", "⚡ Live Simulator",
     "🧾 Transaction Explorer", "📐 Methodology"]
)

# ---------------------------------------------------------------------------
# TAB 1: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    n_retryable = sim_results["n_retryable"].iloc[0]
    naive_rate = (sim_results["naive_recovered_count"] / n_retryable).mean()
    smart_rate = (sim_results["smart_recovered_count"] / n_retryable).mean()
    naive_rev = sim_results["naive_recovered_revenue"].mean()
    smart_rev = sim_results["smart_recovered_revenue"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Naive recovery rate", f"{naive_rate:.1%}")
    col2.metric("Smart recovery rate", f"{smart_rate:.1%}", f"+{(smart_rate-naive_rate)*100:.1f} pp")
    col3.metric("Naive revenue recovered", f"Rs {naive_rev:,.0f}")
    col4.metric("Smart revenue recovered", f"Rs {smart_rev:,.0f}", f"+Rs {smart_rev-naive_rev:,.0f}")

    st.write("")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Naive (blind retry)", "Smart Recovery Engine"],
            y=[naive_rate * 100, smart_rate * 100],
            marker_color=[DANGER, SUCCESS],
            text=[f"{naive_rate:.1%}", f"{smart_rate:.1%}"],
            textposition="outside",
        ))
        fig.update_layout(title="Recovery rate comparison", yaxis_title="Recovery rate (%)",
                           yaxis_range=[0, 100], height=380, showlegend=False)
        apply_dark_theme(fig)
        st.plotly_chart(fig, width='stretch')

    with col_right:
        action_counts = scheduled["action"].value_counts()
        colors_map = {"smart_retry_scheduled": SUCCESS, "whatsapp_nudge": PRIMARY,
                      "escalate_update_payment_method": DANGER,
                      "exclude_route_to_winback": TEXT_MUTED, "none": "#334155"}
        fig2 = go.Figure(data=[go.Pie(
            labels=action_counts.index, values=action_counts.values, hole=0.45,
            marker_colors=[colors_map.get(a, "#94A3B8") for a in action_counts.index],
        )])
        fig2.update_layout(title="What the engine decided (all 10,000 transactions)", height=380)
        apply_dark_theme(fig2)
        st.plotly_chart(fig2, width='stretch')

    

# ---------------------------------------------------------------------------
# TAB 2: Decline Insights
# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader("Not all failures are worth retrying the same way")

    failed = transactions[transactions["initial_status"] == "failed"]
    retried = failed[failed["is_voluntary_churn"] == False]

    success_by_code = retried.groupby("decline_code")["retry_success"].agg(["mean", "count"]).sort_values("mean")
    colors = [DANGER if v < 0.15 else WARNING if v < 0.5 else SUCCESS for v in success_by_code["mean"]]

    fig = go.Figure(go.Bar(
        x=success_by_code["mean"] * 100, y=success_by_code.index, orientation="h",
        marker_color=colors,
        text=[f"{v:.0%} (n={n})" for v, n in zip(success_by_code["mean"], success_by_code["count"])],
        textposition="outside",
    ))
    fig.update_layout(title="Retry success rate by decline code (naive blind retry)",
                       xaxis_title="Success rate (%)", height=450, xaxis_range=[0, 100])
    apply_dark_theme(fig)
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 Not worth blind retry** (< 15% success)")
        for code in success_by_code[success_by_code["mean"] < 0.15].index:
            st.caption(f"- `{code}` — retrying wastes a network retry attempt")
    with col2:
        st.markdown("**🟢 High recovery potential** (> 50% success)")
        for code in success_by_code[success_by_code["mean"] >= 0.5].index:
            st.caption(f"- `{code}` — worth timing intelligently, not just retrying blindly")

    st.divider()
    st.subheader("The salary-window effect (India-specific)")
    funds_related = retried[retried["decline_code"].isin(["INSUFFICIENT_FUNDS", "DO_NOT_HONOR_TEMP"])]
    by_window = funds_related.groupby("is_salary_window")["retry_success"].mean()
    fig3 = go.Figure(go.Bar(
        x=["Outside salary window", "Inside salary window (1st-3rd, 7th-8th)"],
        y=[by_window.get(False, 0) * 100, by_window.get(True, 0) * 100],
        marker_color=[TEXT_MUTED, PRIMARY],
        text=[f"{by_window.get(False,0):.1%}", f"{by_window.get(True,0):.1%}"],
        textposition="outside",
    ))
    fig3.update_layout(title="Funds-related retry success: timing matters",
                        yaxis_title="Success rate (%)", height=380, yaxis_range=[0, 100])
    apply_dark_theme(fig3)
    st.plotly_chart(fig3, width='stretch')

# ---------------------------------------------------------------------------
# TAB 3: Live Simulator
# ---------------------------------------------------------------------------
with tab_simulator:
    st.subheader("⚡ Try it yourself: what would the engine decide?")
    st.caption("Pick a scenario below and watch the engine reason through it live - the same logic that runs on all 10,000 transactions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        payment_method = st.radio("Payment method", ["card", "upi"], horizontal=True)
    with col2:
        available_codes = CARD_CODES if payment_method == "card" else UPI_CODES
        decline_code = st.selectbox("Decline reason", available_codes)
    with col3:
        charge_day = st.slider("Day of month charge failed", 1, 28, 15)

    subscription_amount = st.select_slider(
        "Subscription amount (Rs)", options=[199, 299, 499, 799, 999, 1499], value=799
    )

    if st.button("Run the Smart Recovery Engine ⚡", type="primary"):
        entry = taxonomy_by_code[decline_code]
        decline_category = entry["category"]

        st.write("")
        if decline_code == "UPI_MANDATE_REVOKED":
            st.markdown('<span class="badge badge-exclude">EXCLUDED - VOLUNTARY CHURN</span>', unsafe_allow_html=True)
            st.write("The customer actively cancelled this mandate. Retrying won't help - "
                     "this gets routed to a win-back flow instead of the recovery funnel.")

        elif decline_code in HARD_OVERRIDE_CODES:
            st.markdown('<span class="badge badge-escalate">ESCALATE - UPDATE PAYMENT METHOD</span>', unsafe_allow_html=True)
            st.write(f"**Rule override:** `{decline_code}` has a historical retry success rate of "
                     f"only {entry['base_retry_success_prob']:.0%} - this is a hard rule, "
                     f"not a model decision. No amount of clever timing fixes an expired or blocked card.")
            st.caption(f"Assumed escalation conversion rate: {ESCALATION_SUCCESS_ASSUMPTION:.0%} "
                       f"(documented assumption, see Methodology tab)")

        elif decline_code in NUDGE_INSTEAD_OF_RETRY:
            st.markdown('<span class="badge badge-nudge">WHATSAPP NUDGE</span>', unsafe_allow_html=True)
            st.write("The mandate exists but wasn't confirmed in time - the fix is getting the "
                     "customer to open their UPI app, not blindly retrying the same charge.")
            st.caption(f"Assumed nudge conversion rate: {NUDGE_SUCCESS_ASSUMPTION:.0%} "
                       f"(documented assumption, see Methodology tab)")

        else:
            with st.spinner("Searching candidate retry days..."):
                best_day, best_prob = find_best_retry_day(
                    model_bundle, decline_code, decline_category,
                    payment_method == "upi", charge_day, subscription_amount
                )

            if best_prob >= RETRY_THRESHOLD:
                st.markdown('<span class="badge badge-retry">SMART RETRY SCHEDULED</span>', unsafe_allow_html=True)
                is_salary = best_day in SALARY_WINDOW_DAYS
                st.write(f"**Best retry day: day {best_day}** of the month "
                         f"{'🎯 (inside salary-credit window)' if is_salary else ''}")
                st.write(f"**Model's predicted success probability: {best_prob:.1%}**")

                naive_day = ((charge_day - 1 + 2) % 28) + 1  # naive: fixed ~2 day retry
                st.caption(f"Naive comparison: a blind system would've retried on day {naive_day} "
                           f"regardless of decline reason or timing.")
            else:
                st.markdown('<span class="badge badge-escalate">ESCALATE - LOW PREDICTED SUCCESS</span>', unsafe_allow_html=True)
                st.write(f"Even on the best available day, predicted success is only {best_prob:.1%} - "
                         f"below the {RETRY_THRESHOLD:.0%} threshold. Retrying would likely waste the attempt, "
                         f"so the engine escalates instead.")

# ---------------------------------------------------------------------------
# TAB 4: Transaction Explorer
# ---------------------------------------------------------------------------
with tab_explorer:
    st.subheader("🧾 Explore real transactions from the dataset")

    merged = transactions.merge(
        scheduled[["transaction_id", "action", "scheduled_retry_day", "predicted_success_prob"]],
        on="transaction_id", how="left"
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        action_filter = st.multiselect(
            "Filter by action",
            options=merged["action"].dropna().unique().tolist(),
            default=["smart_retry_scheduled", "whatsapp_nudge", "escalate_update_payment_method"]
        )
    with col2:
        search_id = st.text_input("Or search a specific transaction ID (e.g. txn_000017)")

    filtered = merged[merged["action"].isin(action_filter)]
    if search_id:
        filtered = merged[merged["transaction_id"].str.contains(search_id, case=False, na=False)]

    st.caption(f"Showing {len(filtered):,} of {len(merged):,} transactions")
    st.dataframe(
        filtered[["transaction_id", "payment_method", "decline_code", "decline_category",
                  "charge_day_of_month", "action", "scheduled_retry_day", "predicted_success_prob"]].head(200),
        width='stretch', height=400
    )

# ---------------------------------------------------------------------------
# TAB 5: Methodology
# ---------------------------------------------------------------------------
with tab_methodology:
    st.subheader("📐 Methodology, and what's measured vs. assumed")

    st.markdown("""
    **Why synthetic data?** Real transaction-level retry logs aren't publicly
    available - this project generates 10,000 synthetic transactions calibrated
    against published industry benchmarks (10-15% first-attempt failure rate,
    20-40% involuntary churn share), documented in `data/decline_codes.json`.

    **The model:** Logistic Regression, trained on 1,202 retryable failed
    transactions, achieving ROC-AUC 0.739 - meaning it correctly ranks a
    successful retry above a failed one about 74% of the time.
    """)

    st.markdown('<div class="assumption-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **⚠️ Two numbers in this dashboard are documented assumptions, not measured data:**
    - WhatsApp nudge conversion rate: **{NUDGE_SUCCESS_ASSUMPTION:.0%}**
    - Escalation (update payment method) conversion rate: **{ESCALATION_SUCCESS_ASSUMPTION:.0%}**

    No public dataset gives an exact figure for either. Both were set conservatively
    rather than chosen to inflate the headline result. Real A/B testing against
    live traffic would be needed to validate or correct these specifically.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    **Full day-by-day build log** (including two real bugs found and fixed) is
    available in `docs/day1_notes.md` through `docs/day6_notes.md` on GitHub.
    """)
