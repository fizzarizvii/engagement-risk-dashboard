"""
University of Chester — Student Engagement Risk Dashboard (prototype).

Synthetic demonstration data only. Uses the same v2 composite logic as
``risk_scoring_engine.py`` for scores and component explanations.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure risk engine import works when running: streamlit run app.py
_DASH = Path(__file__).resolve().parent
_ROOT = _DASH.parent
if str(_ROOT / "risk-scoring-engine") not in sys.path:
    sys.path.insert(0, str(_ROOT / "risk-scoring-engine"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.data import (
    component_explanations_for_row,
    enrich_with_recomputed_risk,
    load_assessments,
    load_interventions,
    load_students,
    row_dict_for_engine,
)
from risk_scoring_engine import compute_risk_from_dataframe_row

# --- University of Chester palette (from chester.ac.uk main.css) ---
CH_RED = "#E2231A"
CH_CHARCOAL = "#2E2E27"
CH_GREY = "#E6E6E6"
CH_MUTED = "#757575"
CH_YELLOW = "#F4C023"
CH_PAGE = "#FFFFFF"
CH_PANEL = "#F7F7F6"

RISK_COLORS = {"Low": "#1B7F3B", "Medium": "#C77800", "High": CH_RED}


def _truthy(v: object) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "t")

st.set_page_config(
    page_title="Engagement Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_theme_css() -> None:
    st.markdown(
        f"""
        <style>
            .block-container {{
                padding-top: 1.25rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }}
            header[data-testid="stHeader"] {{
                background: linear-gradient(90deg, {CH_RED} 0%, {CH_RED} 4px, {CH_PAGE} 4px, {CH_PAGE} 100%);
            }}
            div[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {CH_PANEL} 0%, {CH_PAGE} 28%);
                border-right: 1px solid {CH_GREY};
            }}
            .chester-hero {{
                font-family: system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                color: {CH_CHARCOAL};
                border-left: 4px solid {CH_RED};
                padding: 1rem 1.25rem;
                background: {CH_PANEL};
                border-radius: 0 8px 8px 0;
                margin-bottom: 1.25rem;
            }}
            .chester-hero h1 {{
                font-size: 1.55rem;
                font-weight: 650;
                letter-spacing: -0.02em;
                margin: 0 0 0.35rem 0;
            }}
            .chester-hero p {{
                margin: 0;
                color: {CH_MUTED};
                font-size: 0.95rem;
                line-height: 1.45;
            }}
            .metric-card {{
                background: {CH_PAGE};
                border: 1px solid {CH_GREY};
                border-radius: 10px;
                padding: 1rem 1.1rem;
                box-shadow: 0 1px 2px rgba(46,46,39,0.06);
            }}
            span[data-testid="stMetricValue"] {{
                color: {CH_CHARCOAL} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def students_enriched() -> pd.DataFrame:
    return enrich_with_recomputed_risk(load_students())


def plotly_risk_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color=[RISK_COLORS[k] for k in counts.index],
                text=counts.values.tolist(),
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor=CH_PAGE,
        plot_bgcolor=CH_PAGE,
        font=dict(color=CH_CHARCOAL, family="system-ui, sans-serif"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=360,
        yaxis=dict(title="Students", gridcolor=CH_GREY, zeroline=False),
        xaxis=dict(title="Risk band"),
        showlegend=False,
        title=dict(text="Students by risk band", font=dict(size=16)),
    )
    return fig


def plotly_component_stress(names: list[str], values: list[float]) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=CH_RED, opacity=0.88),
        )
    )
    fig.update_layout(
        paper_bgcolor=CH_PAGE,
        plot_bgcolor=CH_PAGE,
        font=dict(color=CH_CHARCOAL, family="system-ui, sans-serif"),
        margin=dict(l=200, r=24, t=32, b=40),
        height=max(320, len(names) * 36),
        xaxis=dict(title="Stress (0–100)", range=[0, 100], gridcolor=CH_GREY),
        yaxis=dict(autorange="reversed"),
        title=dict(text="Model components (higher = more concern)", font=dict(size=15)),
    )
    return fig


def page_overview(df: pd.DataFrame) -> None:
    st.markdown(
        f"""
        <div class="chester-hero">
            <h1>Engagement risk overview</h1>
            <p>Headline metrics and distribution for the simulated cohort. Figures update from the dataset automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n = len(df)
    c_low = int((df["risk_level"] == "Low").sum())
    c_med = int((df["risk_level"] == "Medium").sum())
    c_high = int((df["risk_level"] == "High").sum())
    mean_score = float(df["risk_score"].mean())

    brief_read = df["accessed_upcoming_assessment_brief"].apply(_truthy)
    urgent = df[(df["days_to_nearest_assessment"] <= 14) & (~brief_read)]
    n_urgent = len(urgent)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Students in cohort", f"{n:,}")
    with m2:
        st.metric("Mean risk score", f"{mean_score:.1f}")
    with m3:
        st.metric("High risk", c_high, help="Band High (composite v2)")
    with m4:
        st.metric(
            "Assessment prep gap (≤14 days)",
            n_urgent,
            help="No brief access with nearest assessment within 14 days",
        )

    c1, c2 = st.columns((1.15, 1))
    with c1:
        st.plotly_chart(plotly_risk_distribution(df), use_container_width=True)
    with c2:
        diff = (df["risk_score"] - df["risk_recomputed"]).abs()
        st.markdown(
            f"""
**Engine check:** Recalculated scores match stored scores within **{float(diff.max()):.0f}** max absolute delta
(mean **{float(diff.mean()):.2f}**). Same Python model as ``risk_scoring_engine.py``.

**Band split:** Low **{c_low}** · Medium **{c_med}** · High **{c_high}**
            """
        )
        hist = px.histogram(
            df,
            x="risk_score",
            nbins=25,
            color_discrete_sequence=[CH_RED],
        )
        hist.update_layout(
            paper_bgcolor=CH_PAGE,
            plot_bgcolor=CH_PAGE,
            font=dict(color=CH_CHARCOAL),
            height=280,
            margin=dict(l=40, r=20, t=40, b=40),
            showlegend=False,
            title=dict(text="Risk score distribution", font=dict(size=15)),
            xaxis=dict(title="Risk score", gridcolor=CH_GREY),
            yaxis=dict(title="Count", gridcolor=CH_GREY),
        )
        st.plotly_chart(hist, use_container_width=True)


def page_students(df: pd.DataFrame) -> None:
    st.markdown(
        f"""
        <div class="chester-hero">
            <h1>Student roster</h1>
            <p>Filter and inspect engagement signals. Detail panel uses live component stresses from the risk engine.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prog = sorted(df["program"].dropna().unique().tolist())
    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        q = st.text_input("Search name or ID", "")
    with col_b:
        band = st.multiselect("Risk band", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    with col_c:
        programs = st.multiselect("Programme", prog, default=prog)

    view = df[df["risk_level"].isin(band) & df["program"].isin(programs)]
    if q.strip():
        qq = q.strip().lower()
        view = view[
            view["student_id"].str.lower().str.contains(qq, na=False)
            | view["student_name"].str.lower().str.contains(qq, na=False)
        ]

    sort_key = st.selectbox(
        "Sort by",
        ["risk_score (desc)", "risk_score (asc)", "days_to_nearest_assessment", "student_name"],
    )
    if sort_key == "risk_score (desc)":
        view = view.sort_values("risk_score", ascending=False)
    elif sort_key == "risk_score (asc)":
        view = view.sort_values("risk_score", ascending=True)
    elif sort_key == "days_to_nearest_assessment":
        view = view.sort_values("days_to_nearest_assessment", ascending=True)
    else:
        view = view.sort_values("student_name")

    show = view[
        [
            "student_id",
            "student_name",
            "program",
            "year_of_study",
            "risk_score",
            "risk_level",
            "risk_score_pre_intervention",
            "days_to_nearest_assessment",
            "accessed_upcoming_assessment_brief",
            "engagement_trend",
            "attendance_rate",
            "vle_logins_last_week",
        ]
    ].copy()

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        height=min(520, 40 + len(show) * 35),
    )

    ids = view["student_id"].tolist()
    if not ids:
        st.warning("No rows match your filters.")
        return

    sid = st.selectbox("Select student for detail", ids, index=0)
    row = df[df["student_id"] == sid].iloc[0]
    rd = row_dict_for_engine(row)
    out = compute_risk_from_dataframe_row(pd.Series(rd))
    expl = component_explanations_for_row(row)

    st.subheader(row["student_name"])
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Risk score", int(row["risk_score"]), help="Stored in dataset")
    with k2:
        st.metric("Band", row["risk_level"])
    with k3:
        st.metric("Pre-intervention score", int(out["risk_score_pre_intervention"]))
    with k4:
        st.metric("Intervention history stress", f"{out['intervention_history_stress']:.0f}")

    st.caption(
        f"Recalculated score: **{out['risk_score']}** · Email: {row.get('email', '—')} · "
        f"Commuter: {row['is_commuter']} · Part-time work: {row['works_part_time']}"
    )

    d1, d2 = st.columns((1.1, 1))
    with d1:
        names = [a for a, _ in expl]
        vals = [float(b) for _, b in expl]
        st.plotly_chart(plotly_component_stress(names, vals), use_container_width=True)
    with d2:
        st.markdown("**Engagement snapshot**")
        st.write(
            {
                "Attendance %": f"{float(row['attendance_rate']):.1f}",
                "VLE logins (week)": int(row["vle_logins_last_week"]),
                "VLE hours (week)": f"{float(row['vle_time_hours_last_week']):.1f}",
                "Last grade": int(row["last_assessment_grade"]),
                "GPA (prev.)": f"{float(row['previous_semester_gpa']):.2f}",
                "Trend": row["engagement_trend"],
                "Days to assessment": int(row["days_to_nearest_assessment"]),
                "Accessed brief": row["accessed_upcoming_assessment_brief"],
                "Prior interventions": int(row["prior_intervention_count"]),
            }
        )


def page_interventions(students: pd.DataFrame) -> None:
    st.markdown(
        f"""
        <div class="chester-hero">
            <h1>Interventions</h1>
            <p>Simulated outreach log merged with student records — for outcome storytelling only.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    inv = load_interventions()
    merged = inv.merge(
        students[["student_id", "student_name", "program", "risk_level"]],
        on="student_id",
        how="left",
    )
    st.dataframe(merged, use_container_width=True, hide_index=True, height=400)

    responded = merged["student_responded"].apply(_truthy)
    rate = float(responded.mean()) if len(merged) else 0.0
    improved = merged["engagement_improved"].apply(_truthy)
    st.metric("Recorded response rate", f"{100 * rate:.1f}%")
    st.metric("Flagged engagement improved", f"{100 * float(improved.mean()):.1f}%" if len(merged) else "—")

    by_type = merged.groupby("intervention_type").size().reset_index(name="count")
    fig = px.bar(
        by_type,
        x="intervention_type",
        y="count",
        color_discrete_sequence=[CH_RED],
    )
    fig.update_layout(
        paper_bgcolor=CH_PAGE,
        plot_bgcolor=CH_PAGE,
        font=dict(color=CH_CHARCOAL),
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text="Interventions by channel", font=dict(size=15)),
        xaxis=dict(gridcolor=CH_GREY),
        yaxis=dict(gridcolor=CH_GREY),
    )
    st.plotly_chart(fig, use_container_width=True)


def page_assessments() -> None:
    st.markdown(
        f"""
        <div class="chester-hero">
            <h1>Assessments (cohort)</h1>
            <p>Synthetic module assessments — brief access counts are illustrative.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a = load_assessments()
    a = a.copy()
    if "students_enrolled" in a.columns and a["students_enrolled"].notna().any():
        a["brief_access_rate"] = (
            100.0 * a["students_accessed_brief"].astype(float) / a["students_enrolled"].astype(float)
        ).round(1)
    st.dataframe(a, use_container_width=True, hide_index=True)

    if "brief_access_rate" in a.columns:
        fig = px.bar(
            a,
            x="assessment_title",
            y="brief_access_rate",
            color_discrete_sequence=[CH_CHARCOAL],
        )
        fig.update_layout(
            paper_bgcolor=CH_PAGE,
            plot_bgcolor=CH_PAGE,
            font=dict(color=CH_CHARCOAL),
            height=400,
            margin=dict(l=40, r=20, t=40, b=120),
            title=dict(text="Brief access rate by assessment", font=dict(size=15)),
            xaxis=dict(tickangle=-35, gridcolor=CH_GREY),
            yaxis=dict(title="% enrolled", gridcolor=CH_GREY, range=[0, 100]),
        )
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    _inject_theme_css()
    df = students_enriched()

    with st.sidebar:
        st.markdown(
            f'<p style="color:{CH_CHARCOAL};font-weight:700;font-size:1.05rem;margin-bottom:0.2rem;">'
            f"University of Chester</p>",
            unsafe_allow_html=True,
        )
        st.caption("Student engagement risk · staff prototype")
        st.markdown("---")
        page = st.radio(
            "Section",
            ["Overview", "Students", "Interventions", "Assessments"],
        )
        st.markdown("---")
        st.markdown(
            f'<p style="font-size:0.8rem;color:{CH_MUTED};line-height:1.4;">'
            "<strong>Demonstration only.</strong> Data are synthetic. "
            "Not for decisions about real students. "
            "Risk scores use composite model v2 from the project engine.</p>",
            unsafe_allow_html=True,
        )

    if page == "Overview":
        page_overview(df)
    elif page == "Students":
        page_students(df)
    elif page == "Interventions":
        page_interventions(df)
    else:
        page_assessments()


if __name__ == "__main__":
    main()
