"""SleepSense: stress-level prediction and personalised what-if coach."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
AUTHOR = "Harshil R"          # <- change this to the name shown in the app
COURSE = "Healthcare Data Analytics"
# ----------------------------------------------------------------------------

ROOT = Path(__file__).parent
TEAL, CYAN, AMBER, ROSE, SLATE = "#2DD4BF", "#38BDF8", "#FBBF24", "#FB7185", "#94A3B8"

st.set_page_config(page_title="SleepSense | Stress Predictor", page_icon="🌙", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1180px;}
.hero {background: linear-gradient(120deg,#0F2A43 0%,#134E4A 100%); border:1px solid #1E3A5F;
       border-radius: 18px; padding: 28px 32px; margin-bottom: 18px;}
.hero h1 {margin:0; font-size: 2.1rem; letter-spacing:-0.5px;}
.hero p {margin:6px 0 0 0; color:#B6C7D8; font-size:1.02rem;}
.card {background:#131C2E; border:1px solid #22304A; border-radius:14px; padding:18px 20px; height:100%;}
.card h4 {margin:0 0 6px 0; color:#2DD4BF; font-size:0.95rem; text-transform:uppercase; letter-spacing:1.2px;}
.card .big {font-size:2rem; font-weight:700; line-height:1.1;}
.card .sub {color:#94A3B8; font-size:0.9rem; margin-top:4px;}
.pill {display:inline-block; padding:3px 12px; border-radius:99px; font-weight:600; font-size:0.85rem;}
.tip {background:#10241F; border:1px solid #1D4D43; border-radius:12px; padding:12px 16px; margin-bottom:10px;}
.tip b {color:#2DD4BF;}
.small {color:#94A3B8; font-size:0.85rem;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ------------------------------- data / model --------------------------------
@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "stress_model.joblib")


@st.cache_data
def load_metrics():
    return json.loads((ROOT / "models" / "metrics.json").read_text())


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "sleep_stress_data.csv")


model, M, DF = load_model(), load_metrics(), load_data()
FEATURES = ["Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep",
            "Physical Activity Level", "BMI Category", "Heart Rate", "Daily Steps",
            "Sleep Disorder", "BP High", "BP Low"]
R = M["feature_ranges"]


def predict(profile: dict) -> float:
    x = pd.DataFrame([profile])[FEATURES]
    return float(np.clip(model.predict(x)[0], 1, 10))


def band(score: float):
    if score <= 4.5:
        return "Low", TEAL
    if score <= 6.5:
        return "Moderate", AMBER
    return "High", ROSE


def gauge(score: float, title: str = "Predicted stress level"):
    label, color = band(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score, 2), number={"font": {"size": 54, "color": color}},
        title={"text": f"{title}<br><span style='font-size:0.95em;color:{color}'>{label}</span>",
               "font": {"size": 16, "color": "#CBD5E1"}},
        gauge={"axis": {"range": [1, 10], "tickcolor": SLATE, "tickfont": {"color": SLATE}},
               "bar": {"color": color, "thickness": 0.28}, "bgcolor": "#131C2E", "borderwidth": 0,
               "steps": [{"range": [1, 4.5], "color": "#123C3A"},
                         {"range": [4.5, 6.5], "color": "#3F3414"},
                         {"range": [6.5, 10], "color": "#431F2B"}]}))
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=70, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#E2E8F0"})
    return fig


def style(fig, h=360):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=40, b=10), font={"color": "#E2E8F0"})
    return fig


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def card(head, big, sub="", color="#E2E8F0"):
    return (f'<div class="card"><h4>{head}</h4><div class="big" style="color:{color}">{big}</div>'
            f'<div class="sub">{sub}</div></div>')


# ------------------------------- navigation ----------------------------------
with st.sidebar:
    st.markdown("## 🌙 SleepSense")
    st.caption("Stress-level prediction from sleep, health and lifestyle data")
    page = st.radio("Navigate", ["Home", "Stress Coach", "Model Lab", "Data Explorer", "About"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Built by **{AUTHOR}**  \n{COURSE}")

res = pd.DataFrame(M["results"])
best = M["best_model"]
best_row = res[(res["model"] == best) & (res["data"] == "real only")].iloc[0]


# ================================== HOME =====================================
if page == "Home":
    hero("Can your sleep predict your stress?",
         "A machine-learning app trained on sleep, health and lifestyle data, with a personal "
         "coach that shows what you could change to lower your predicted stress.")
    c = st.columns(4)
    c[0].markdown(card("Training data", f"{M['n_total']:,}", f"real records, {M['n_unique_original']} unique profiles"), unsafe_allow_html=True)
    c[1].markdown(card("Model", "Blend", "ElasticNet + Gradient Boosting"), unsafe_allow_html=True)
    c[2].markdown(card("R² on unseen people", f"{best_row['r2_mean']:.2f}", "real held-out profiles", TEAL), unsafe_allow_html=True)
    c[3].markdown(card("Typical error", f"±{best_row['rmse_mean']:.2f}", "stress points (scale 3–8)", CYAN), unsafe_allow_html=True)

    st.markdown("### What's inside")
    a, b, d = st.columns(3)
    a.markdown(card("Stress Coach ✨", "What-if simulator",
                    "Predict your stress, see which habits drive it, and test a personal improvement plan."),
               unsafe_allow_html=True)
    b.markdown(card("Model Lab", "Regularisation",
                    "Linear vs Ridge, Lasso, ElasticNet and tree models, scored on real unseen data."),
               unsafe_allow_html=True)
    d.markdown(card("Data Explorer", f"{M['n_total']} rows",
                    "Browse, filter and download the dataset behind the model."),
               unsafe_allow_html=True)

    st.markdown("### Strongest signals in the data")
    corr = DF[["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Heart Rate",
               "Daily Steps", "BP High", "BP Low", "Stress Level"]].corr()["Stress Level"].drop("Stress Level").sort_values()
    fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h",
                           marker_color=[TEAL if v < 0 else ROSE for v in corr.values],
                           text=[f"{v:+.2f}" for v in corr.values], textposition="outside"))
    fig.update_layout(title="Correlation with stress level (teal: protective, rose: risk)", xaxis_range=[-1.1, 1.1])
    st.plotly_chart(style(fig, 340), width='stretch')
    st.info("This is an educational project, not a medical tool. Results describe patterns in a "
            "dataset, not a diagnosis.")


# ============================== STRESS COACH =================================
elif page == "Stress Coach":
    hero("Stress Coach", "Enter your profile, get a predicted stress level, then explore what would lower it.")

    with st.expander("How does Stress Coach work?"):
        st.markdown("""
1. **Predict.** Your 12 inputs go through the trained model (a blend of ElasticNet regression and Gradient Boosting) to give a stress score.
2. **Test each habit.** For sleep duration, sleep quality, activity and steps, the app copies your profile, changes *only that one habit* to a healthy target (8 h, quality 8, 75 min, 8,000 steps), and predicts again. The difference is that habit's possible reduction.
3. **Rank.** Habits are ranked by how much they would lower your score, giving the top recommendations.
4. **What-if plan.** The sliders change several habits at once and the model re-predicts instantly, so you can compare *now* with *plan*.

These are associations learned from the dataset, not proven cause and effect, and the tool is not medical advice.
""")

    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("#### Your profile")
        g1, g2 = st.columns(2)
        gender = g1.selectbox("Gender", M["categories"]["Gender"])
        age = g2.slider("Age", int(R["Age"][0]), int(R["Age"][1]), 35)
        occupation = st.selectbox("Occupation", M["categories"]["Occupation"],
                                  index=M["categories"]["Occupation"].index("Nurse"))
        bmi = g1.selectbox("BMI category", M["categories"]["BMI Category"])
        disorder = g2.selectbox("Sleep disorder", M["categories"]["Sleep Disorder"],
                                index=M["categories"]["Sleep Disorder"].index("No Disorder"))
        sleep = st.slider("Sleep duration (hours)", R["Sleep Duration"][0], R["Sleep Duration"][1], 6.8, 0.1)
        quality = st.slider("Sleep quality (1–10)", int(R["Quality of Sleep"][0]), int(R["Quality of Sleep"][1]), 6)
        activity = st.slider("Physical activity (min/day)", int(R["Physical Activity Level"][0]),
                             int(R["Physical Activity Level"][1]), 45, 5)
        steps = st.slider("Daily steps", int(R["Daily Steps"][0]), int(R["Daily Steps"][1]), 5500, 100)
        h1, h2, h3 = st.columns(3)
        hr = h1.number_input("Resting HR", int(R["Heart Rate"][0]), int(R["Heart Rate"][1]), 72)
        bph = h2.number_input("BP high", int(R["BP High"][0]), int(R["BP High"][1]), 125)
        bpl = h3.number_input("BP low", int(R["BP Low"][0]), int(R["BP Low"][1]), 82)
        st.caption("Inputs are limited to the ranges seen in the training data.")

    profile = {"Gender": gender, "Age": age, "Occupation": occupation, "Sleep Duration": sleep,
               "Quality of Sleep": quality, "Physical Activity Level": activity, "BMI Category": bmi,
               "Heart Rate": hr, "Daily Steps": steps, "Sleep Disorder": disorder,
               "BP High": bph, "BP Low": bpl}
    score = predict(profile)
    label, color = band(score)

    with right:
        st.plotly_chart(gauge(score), width='stretch')
        avg = DF["Stress Level"].mean()
        diff = score - avg
        st.markdown(
            f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}">'
            f'{label} stress</span> &nbsp; <span class="small">dataset average is {avg:.1f}; '
            f'you are {abs(diff):.1f} {"above" if diff > 0 else "below"}.</span>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔍 What drives your score", "🧪 What-if plan", "📄 Report"])

    # levers: (feature, healthy target, label, unit text)
    LEVERS = [("Sleep Duration", 8.0, "Sleep 8 hours", "h"),
              ("Quality of Sleep", 8, "Raise sleep quality to 8", ""),
              ("Physical Activity Level", 75, "Reach 75 min activity/day", " min"),
              ("Daily Steps", 8000, "Reach 8,000 steps/day", " steps")]
    gains = []
    for feat, target, text, _ in LEVERS:
        if profile[feat] < target:
            alt = dict(profile, **{feat: target})
            gains.append((text, score - predict(alt)))
        else:
            gains.append((text + " (already met)", 0.0))
    gains.sort(key=lambda t: t[1])

    with tab1:
        st.markdown("Each bar shows how much your predicted stress would drop if **only that one habit** "
                    "moved to a healthy target.")
        fig = go.Figure(go.Bar(x=[g for _, g in gains], y=[t for t, _ in gains], orientation="h",
                               marker_color=[TEAL if g > 0.01 else "#334155" for _, g in gains],
                               text=[f"-{g:.2f}" if g > 0.01 else "–" for _, g in gains], textposition="outside"))
        fig.update_layout(title="Possible reduction in predicted stress", xaxis_title="Stress points",
                          yaxis={"autorange": "reversed"})
        fig.update_xaxes(range=[0, max(0.5, max(g for _, g in gains) * 1.35)])
        st.plotly_chart(style(fig, 320), width='stretch')
        top = [(t, g) for t, g in gains if g > 0.01]
        top.sort(key=lambda t: -t[1])
        if top:
            for t, g in top[:3]:
                st.markdown(f'<div class="tip"><b>{t}</b>: predicted stress drops by about <b>{g:.2f}</b>.</div>',
                            unsafe_allow_html=True)
        else:
            st.success("All four lifestyle levers already meet the healthy targets.")

    with tab2:
        st.markdown("Adjust the four habits you can change and compare against your current profile.")
        w1, w2 = st.columns(2)
        n_sleep = w1.slider("New sleep duration (h)", R["Sleep Duration"][0], R["Sleep Duration"][1], float(min(max(sleep, 7.5), R["Sleep Duration"][1])), 0.1, key="ns")
        n_quality = w2.slider("New sleep quality", int(R["Quality of Sleep"][0]), int(R["Quality of Sleep"][1]), int(max(quality, 8)), key="nq")
        n_act = w1.slider("New activity (min/day)", int(R["Physical Activity Level"][0]), int(R["Physical Activity Level"][1]), int(max(activity, 60)), 5, key="na")
        n_steps = w2.slider("New daily steps", int(R["Daily Steps"][0]), int(R["Daily Steps"][1]), int(max(steps, 7500)), 100, key="nst")
        plan = dict(profile, **{"Sleep Duration": n_sleep, "Quality of Sleep": n_quality,
                                "Physical Activity Level": n_act, "Daily Steps": n_steps})
        new = predict(plan)
        delta = new - score
        a, b, c3 = st.columns(3)
        a.metric("Current", f"{score:.2f}", band(score)[0])
        b.metric("With your plan", f"{new:.2f}", f"{delta:+.2f}", delta_color="inverse")
        c3.metric("Category", band(new)[0], f"was {band(score)[0]}", delta_color="off")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Current", "With plan"], y=[score, new],
                             marker_color=[band(score)[1], band(new)[1]],
                             text=[f"{score:.2f}", f"{new:.2f}"], textposition="outside"))
        fig.update_layout(yaxis_range=[0, 10], title="Predicted stress: now vs plan")
        st.plotly_chart(style(fig, 300), width='stretch')

    with tab3:
        lines = [f"SleepSense stress report - {AUTHOR}", "=" * 44,
                 f"Predicted stress level: {score:.2f} / 10  ({label})",
                 f"Dataset average: {DF['Stress Level'].mean():.2f}", "", "Profile:"]
        lines += [f"  {k}: {v}" for k, v in profile.items()]
        lines += ["", "Biggest single-habit improvements:"]
        lines += [f"  {t}: -{g:.2f}" for t, g in sorted(gains, key=lambda t: -t[1]) if g > 0.01] or ["  none (targets met)"]
        lines += ["", "Educational use only. Not medical advice."]
        report = "\n".join(lines)
        st.code(report, language=None)
        st.download_button("Download report (.txt)", report, file_name="stress_report.txt")


# ================================ MODEL LAB ==================================
elif page == "Model Lab":
    hero("Model Lab", "Six model families compared on real, unseen people, including regularised regression.")
    st.markdown(
        f"**How we score fairly.** The dataset has {M['n_original']} records but only "
        f"{M['n_unique_original']} unique people, because many rows repeat. To avoid inflated scores, we split by "
        "unique person and score on people the model never saw during training. Numbers are the mean of 20 random splits.")

    show = res.copy()
    show["R²"] = show["r2_mean"].round(3)
    show["RMSE"] = show["rmse_mean"].round(3)
    show["MAPE %"] = (show["mape_mean"] * 100).round(1)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        fig = px.bar(show, x="model", y="R²", color_discrete_sequence=[TEAL],
                     title="R² on unseen people", labels={"model": "", "data": "Training data"})
        fig.update_layout(yaxis_range=[0.7, 1.0], xaxis_tickangle=-25)
        st.plotly_chart(style(fig, 400), width='stretch')
    with c2:
        st.dataframe(show[["model", "R²", "RMSE", "MAPE %"]]
                     .sort_values("RMSE").set_index("model"), width='stretch')
        st.success(f"Selected model: **{best}**")
        st.caption("The blend averages a regularised linear model (stable, smooth) with gradient boosting "
                   "(captures non-linear effects).")

    st.markdown("### Regularisation at work")
    st.markdown("Ridge (L2) shrinks coefficients, Lasso (L1) can set them exactly to zero, ElasticNet does both. "
                "Plain linear regression has no such control.")
    coef = pd.DataFrame({k: {f: v for f, v in d.items() if f in
                             ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
                              "Heart Rate", "Daily Steps", "BP High", "BP Low"]}
                         for k, d in M["coefficients"].items()})
    fig = px.bar(coef.reset_index().melt("index"), x="index", y="value", color="variable", barmode="group",
                 title="Standardised coefficients by model", labels={"index": "", "value": "Coefficient", "variable": "Model"},
                 color_discrete_sequence=[SLATE, CYAN, TEAL, AMBER])
    st.plotly_chart(style(fig, 380), width='stretch')
    if M["lasso_zeroed"]:
        st.info("Lasso removed these features entirely (coefficient 0): " + ", ".join(M["lasso_zeroed"]))

    st.markdown("### What we learned")
    lin = res[(res["model"] == "Linear Regression") & (res["data"] == "real only")].iloc[0]
    en = res[(res["model"] == "ElasticNet (L1+L2)") & (res["data"] == "real only")].iloc[0]
    st.markdown(
        f"- Regularisation helped: R² rose from **{lin['r2_mean']:.2f}** (plain linear) to **{en['r2_mean']:.2f}** (ElasticNet).\n"
        f"- Blending a regularised model with boosting gave the best score, **R² {best_row['r2_mean']:.2f}**.\n"
        "- More real data is the clearest way to improve the model further: the dataset has only "
        f"{M['n_unique_original']} unique people.")


# ============================== DATA EXPLORER ================================
elif page == "Data Explorer":
    hero("Data Explorer", f"{M['n_total']} records from the Kaggle Sleep Health and Lifestyle dataset.")
    f2, f3 = st.columns(2)
    occ = f2.multiselect("Occupation", M["categories"]["Occupation"])
    dis = f3.multiselect("Sleep disorder", M["categories"]["Sleep Disorder"])
    view = DF.drop(columns="Source")
    if occ:
        view = view[view["Occupation"].isin(occ)]
    if dis:
        view = view[view["Sleep Disorder"].isin(dis)]
    st.caption(f"{len(view):,} rows shown")
    st.dataframe(view, width='stretch', height=300, hide_index=True)
    st.download_button("Download CSV", view.to_csv(index=False), "sleep_stress_dataset.csv")

    a, b = st.columns(2)
    fig = px.histogram(view, x="Stress Level", nbins=6, color_discrete_sequence=[TEAL], title="Stress level distribution")
    a.plotly_chart(style(fig, 320), width='stretch')
    occ_mean = view.groupby("Occupation")["Stress Level"].mean().sort_values().reset_index()
    fig = px.bar(occ_mean, x="Stress Level", y="Occupation", orientation="h", title="Average stress by occupation",
                 color_discrete_sequence=[CYAN])
    b.plotly_chart(style(fig, 320), width='stretch')
    fig = px.scatter(view, x="Quality of Sleep", y="Stress Level", opacity=0.6,
                     color_discrete_sequence=[TEAL], title="Sleep quality vs stress")
    st.plotly_chart(style(fig, 340), width='stretch')


# ================================== ABOUT ====================================
else:
    hero("About this project", f"{COURSE} course project by {AUTHOR}.")
    st.markdown(f"""
**Goal.** Predict a person's stress level (scale 3–8) from 12 sleep, health and lifestyle features.

**Data.** The *Sleep Health and Lifestyle* dataset from Kaggle ({M['n_original']} records, of which {M['n_unique_original']} are unique people).

**Method.**
1. Clean the data (blood pressure split, categories merged), one-hot encode and standardise.
2. Train linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting and a blend.
3. Evaluate on people never used in training (20 random splits grouped by unique person).
4. Deploy the best model with a what-if simulator (Stress Coach) that re-predicts after each habit change.

**Limitations.**
- Small, partly repeated dataset ({M['n_unique_original']} unique people); more real data would help.
- The what-if tool shows statistical associations in this dataset, not proven cause and effect.
- Stress only ranges from 3 to 8 in the data; predictions outside this range are unreliable.
- Not a medical device; not for diagnosis.

**Credits.** Dataset: Kaggle, *Sleep Health and Lifestyle Dataset*. Built with Python, scikit-learn and Streamlit.
""")
