"""WattWise - Household Energy Intelligence (Streamlit dashboard)."""
import os, sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import utils as U

st.set_page_config(page_title="WattWise", page_icon="⚡", layout="wide")

COLORS = {"Normal": "#34d399", "Unusually High": "#fb923c", "Standby-like": "#a78bfa"}
PANEL, GRID, TEXT, MUTED = "#16233b", "#27385a", "#e2e8f0", "#94a3b8"
STEP = {"1x": 5, "5x": 25, "20x": 100}   # readings (minutes of data) per 1-second tick

st.markdown(f"""<style>
.block-container {{padding-top: 4.5rem; max-width: 1400px;}}
[data-testid=stHorizontalBlock] {{gap: 1rem;}}
.ww-gap {{height: 1rem;}}
.ww-title {{font-size: 2.6rem !important; font-weight: 800 !important; margin: 0 !important; line-height: 1.15 !important; color: #22d3ee;}}
.ww-sub {{color: {MUTED}; font-size: 1.05rem !important; margin: 0 0 .4rem 0 !important;}}
.ww-badge {{display:inline-block; padding:.2rem .7rem; border-radius:999px;
  background:#0e3a4a; color:#67e8f9; font-size:.8rem; white-space:nowrap; border:1px solid #155e75;}}
.ww-card {{background:{PANEL}; border:1px solid {GRID}; border-radius:16px; padding:1rem 1.2rem; height:8.6rem; box-sizing:border-box; overflow:hidden;}}
.ww-label {{color:{MUTED}; font-size:.75rem; text-transform:uppercase; letter-spacing:.04em; word-break:keep-all; overflow-wrap:normal;}}
.ww-value {{font-size:1.8rem; font-weight:700; line-height:1.3; white-space:nowrap;}}
.ww-note {{color:{MUTED}; font-size:.78rem;}}
</style>""", unsafe_allow_html=True)


# ---------- cached resources (loaded once per server process) ----------
@st.cache_resource(show_spinner="Loading trained model...")
def get_model():
    return joblib.load(U.ARTIFACT_DIR / "wattwise_rf.joblib")


@st.cache_resource(show_spinner="Reading dataset and building features (first load only)...")
def get_data():
    path = U.find_dataset(os.environ.get("WATTWISE_DATA_DIR"))
    return U.add_features(U.load_raw(path))


@st.cache_data(show_spinner=False)
def get_day(day_iso: str):
    """One day's grid (NaN = missing) plus predictions for its valid readings."""
    d = pd.Timestamp(day_iso)
    grid = get_data().loc[d: d + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)].copy()
    grid["pred"] = None
    v = grid["valid"]
    if v.any():
        grid.loc[v, "pred"] = U.predict(get_model(), grid[v])
    return grid


meta = U.load_metadata()
avail = [date.fromisoformat(d) for d in meta["available_dates"]]
avail_set = set(avail)
demo = meta["demo_days"][0] if meta["demo_days"] else None

# ---------- session state ----------
ss = st.session_state
ss.setdefault("sel_date", date.fromisoformat(demo["date"]) if demo else avail[-1])
ss.setdefault("idx", 0)
ss.setdefault("playing", False)
ss.setdefault("show_all", False)


def reset(playing=False):
    ss.idx, ss.playing = 0, playing


def load_demo():
    if demo:
        ss.sel_date = date.fromisoformat(demo["date"])
    reset()


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### ⚡ Replay controls")
    st.button("🎬 Load Demo Day", on_click=load_demo, width="stretch")
    st.date_input("Date", key="sel_date", min_value=avail[0], max_value=avail[-1],
                  on_change=reset, format="YYYY-MM-DD")
    speed = st.select_slider("Playback speed", ["1x", "5x", "20x"], value="5x")
    st.caption("1x = 5 min of data per second")
    c1, c2 = st.columns(2)
    if c1.button("▶ Start", width="stretch"):
        ss.playing = True
    if c2.button("⏸ Pause", width="stretch"):
        ss.playing = False
    st.button("↺ Reset", on_click=reset, width="stretch")
    st.checkbox("Show full day immediately", key="show_all")
    if demo:
        st.caption(f"Demo day: {demo['date']} — "
                   f"{demo['Normal']} Normal / {demo['Unusually High']} High / "
                   f"{demo['Standby-like']} Standby-like readings (model predictions)")

# ---------- header ----------
h1, h2 = st.columns([2, 1])
with h1:
    st.markdown('<div class="ww-title">⚡ WattWise</div><div class="ww-sub">Household Energy Intelligence</div>',
                unsafe_allow_html=True)
with h2:
    st.markdown('<div style="text-align:right;padding-top:1rem"><span class="ww-badge">'
                '● Historical data (UCI) — not live</span></div>', unsafe_allow_html=True)

tab_dash, tab_model = st.tabs(["Dashboard", "Model Information"])


def card(label, value, note="", color=TEXT, size=None):
    fs = f";font-size:{size}" if size else ""
    return (f'<div class="ww-card"><div class="ww-label">{label}</div>'
            f'<div class="ww-value" style="color:{color}{fs}">{value}</div>'
            f'<div class="ww-note">{note}</div></div>')


def style_fig(fig, height, title=None, legend=True):
    fig.update_layout(height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      font=dict(color=TEXT), showlegend=legend,
                      margin=dict(l=60, r=24, t=60 if title else 20, b=110 if legend else 50),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, y=0.96, yanchor="top",
                                     font=dict(size=15)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def render_replay():
    """Everything that changes during replay. Runs as a fragment so only this
    block refreshes each second while the sidebar stays responsive."""
    day = ss.sel_date
    if day not in avail_set:
        st.warning(f"No readings for {day}. Choose a date between {avail[0]} and {avail[-1]} "
                   "(some dates have no data), or press Load Demo Day.")
        return
    grid = get_day(day.isoformat())
    valid = grid[grid["valid"]]
    n = len(valid)
    if n == 0:
        st.warning("This date has no complete readings (all missing)."); return

    if ss.show_all:
        ss.idx = n - 1
    elif ss.playing:
        ss.idx = min(ss.idx + STEP[speed], n - 1)
    idx = min(ss.idx, n - 1)
    seen = valid.iloc[: idx + 1]
    cur = seen.iloc[-1]
    if idx >= n - 1 and ss.playing:
        ss.playing = False
        st.rerun(scope="app")    # refresh sidebar-driven state once replay finishes

    counts = seen["pred"].value_counts().reindex(U.CLASSES, fill_value=0)
    en = U.energy_kwh(seen["Global_active_power"])
    high_pct = 100 * counts["Unusually High"] / max(len(seen), 1)

    k = st.columns(4)
    k[0].markdown(card("Active Power", f"{cur['Global_active_power']:.2f} kW",
                       f"instantaneous · at {seen.index[-1]:%H:%M}", "#22d3ee"), unsafe_allow_html=True)
    k[1].markdown(card("Usage Classification", cur["pred"], "Random Forest prediction",
                       COLORS[cur["pred"]]), unsafe_allow_html=True)
    miss = f" · {en['n_missing']} missing min excluded" if en["n_missing"] else ""
    k[2].markdown(card("Energy Consumed", f"{en['kwh']:.2f} kWh",
                       f"accumulated so far on {day}{miss}", "#60a5fa"), unsafe_allow_html=True)
    k[3].markdown(card("High Usage", f"{high_pct:.1f} %",
                       "of readings flagged High", "#fb923c"), unsafe_allow_html=True)

    # --- power chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid.index[grid.index <= seen.index[-1]],
                             y=grid["Global_active_power"][grid.index <= seen.index[-1]],
                             mode="lines", line=dict(color="#3b82f6", width=1.5),
                             name="Active power", connectgaps=False))
    for c in U.CLASSES:
        s = seen[seen["pred"] == c]
        if len(s):
            fig.add_trace(go.Scatter(x=s.index, y=s["Global_active_power"], mode="markers",
                                     marker=dict(color=COLORS[c], size=5), name=c))
    fig.add_trace(go.Scatter(x=[seen.index[-1]], y=[cur["Global_active_power"]], mode="markers",
                             marker=dict(size=15, color="rgba(0,0,0,0)",
                                         line=dict(color="#22d3ee", width=3)),
                             name="Latest reading"))
    fig.update_xaxes(range=[grid.index[0], grid.index[-1]], title="Time")
    fig.update_yaxes(title="Active Power (kW)", rangemode="tozero")
    st.markdown('<div class="ww-gap"></div>', unsafe_allow_html=True)
    st.plotly_chart(style_fig(fig, 460, "Power Consumption · Historical Smart Meter Replay"),
                    width="stretch")

    # --- summary + table ---
    left, right = st.columns([1.3, 1.3])
    with left:
        st.markdown("##### Classification Summary")
        m = st.columns(3)
        for col, c in zip(m, U.CLASSES):
            col.markdown(card(c, f"{counts[c]:,}", "readings", COLORS[c]), unsafe_allow_html=True)
        tot = max(int(counts.sum()), 1)
        donut = go.Figure(go.Pie(labels=U.CLASSES, values=[counts[c] for c in U.CLASSES], hole=.62,
                                 marker=dict(colors=[COLORS[c] for c in U.CLASSES]),
                                 text=[f"{100 * counts[c] / tot:.1f}%" if counts[c] else "" for c in U.CLASSES],
                                 textinfo="text", textfont=dict(color="#0b1426"), sort=False))
        st.markdown('<div class="ww-gap"></div>', unsafe_allow_html=True)
        st.plotly_chart(style_fig(donut, 280, legend=False), width="stretch")
    with right:
        st.markdown("##### Recent Readings")
        t = seen.tail(10).iloc[::-1]
        tbl = pd.DataFrame({"Time": t.index.strftime("%d %b %H:%M"),
                            "Power (kW)": t["Global_active_power"].round(3).values,
                            "Volts (V)": t["Voltage"].round(1).values,
                            "Amps (A)": t["Global_intensity"].round(1).values,
                            "Class": t["pred"].values})
        st.dataframe(tbl.style.map(lambda v: f"color:{COLORS.get(v, TEXT)};font-weight:600",
                                   subset=["Class"])
                         .format({"Power (kW)": "{:.3f}", "Volts (V)": "{:.1f}", "Amps (A)": "{:.1f}"}),
                     hide_index=True, width="stretch")
        st.caption(f"Reading {idx + 1} of {n} on {day} · "
                   f"{'▶ playing' if ss.playing else '⏸ paused'}")


with tab_dash:
    if ss.sel_date in avail_set:
        get_data()   # warm cache with a spinner before the fragment starts
    frag = st.fragment(run_every=1.0 if ss.playing else None)(render_replay)
    frag()
    if demo and not demo["in_test"]:
        st.info("Note: no held-out day contained all three classes; demo day is from the training period.")

with tab_model:
    th = meta["thresholds"]
    st.subheader("Model Information")
    st.warning("**Evaluation measures agreement with rule-based pseudo-labels, not verified "
               "real-world consumption states.** The UCI data has no ground-truth labels, and the "
               "labels are computed from power and rolling-variability features the model also "
               "receives, so the model is largely learning a proxy for those rules. It is not a "
               "validated anomaly detector, and “Standby-like” is a low-stable-load pattern, not an "
               "identified appliance.")
    a = st.columns(4)
    a[0].markdown(card("Algorithm", "Random Forest", f"{meta['rf_params']['n_estimators']} trees, "
                       f"max depth {meta['rf_params']['max_depth']}", size="1.5rem"), unsafe_allow_html=True)
    a[1].markdown(card("Accuracy", f"{meta['accuracy']:.4f}", "vs pseudo-labels (chrono holdout)",
                       "#22d3ee"), unsafe_allow_html=True)
    a[2].markdown(card("Macro F1", f"{meta['macro_f1']:.4f}", "vs pseudo-labels", "#22d3ee"),
                  unsafe_allow_html=True)
    a[3].markdown(card("Dataset size", f"{meta['valid_rows']:,}", f"of "
                       f"{meta['grid_minutes']:,} minutes ({meta['missing_minutes']:,} missing)"),
                  unsafe_allow_html=True)
    st.markdown('<div class="ww-gap"></div>', unsafe_allow_html=True)
    st.markdown(f"**Dataset:** {meta['dataset']} (`{meta['file']}`), {meta['first_ts'][:10]} → "
                f"{meta['last_ts'][:10]}.  \n"
                f"**Train:** {meta['train_start'][:10]} → {meta['train_end'][:10]} "
                f"({meta['n_train']:,} rows) · **Test:** {meta['test_start'][:10]} → "
                f"{meta['test_end'][:10]} ({meta['n_test']:,} rows). Chronological split, no shuffling.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Class labeling rules (thresholds from training data only)")
        st.markdown(f"""
- **Unusually High:** active power > **{th['high_power_kw']:.3f} kW** (training {meta['label_params']['HIGH_Q']:.0%} quantile)
- **Standby-like:** active power ≤ **{th['standby_power_kw']:.3f} kW** (training {meta['label_params']['LOW_Q']:.0%} quantile)
  **and** {meta['label_params']['WINDOW']}-min rolling std ≤ **{th['standby_roll_std_kw']:.4f} kW** (training median)
- **Normal:** everything else

Rolling features use the current and past readings only.""")
        st.markdown("##### Features")
        st.markdown(" ".join(f"`{f}`" for f in meta["features"]))
        st.markdown("##### How a Random Forest predicts")
        st.markdown("It trains 100 small decision trees, each on a random sample of the data. "
                    "Each tree asks a chain of yes/no questions about a reading (e.g. “is power above "
                    "3.4 kW?”) and votes for a class. The forest's prediction is the majority vote.")
    with c2:
        cm = meta["confusion_matrix"]
        hm = go.Figure(go.Heatmap(z=cm, x=U.CLASSES, y=U.CLASSES, colorscale="Blues",
                                  text=[[f"{v:,}" for v in r] for r in cm], texttemplate="%{text}",
                                  showscale=False))
        hm.update_xaxes(title="Predicted"); hm.update_yaxes(title="Pseudo-label", autorange="reversed")
        st.plotly_chart(style_fig(hm, 340, "Confusion matrix (test set)", legend=False), width="stretch")
        fi = sorted(meta["feature_importance"].items(), key=lambda x: x[1])
        bar = go.Figure(go.Bar(x=[v for _, v in fi], y=[k for k, _ in fi], orientation="h",
                               marker_color="#22d3ee"))
        st.plotly_chart(style_fig(bar, 340, "Feature importance", legend=False), width="stretch")
    rep = pd.DataFrame(meta["report"]).T.loc[U.CLASSES + ["macro avg"]]
    st.markdown("##### Classification report (test set)")
    st.dataframe(rep.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}",
                                   "support": "{:,.0f}"}), width="stretch")
