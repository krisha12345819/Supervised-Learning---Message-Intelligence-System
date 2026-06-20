"""
==========================================================================================
 MESSAGE INTELLIGENCE SYSTEM
 An interactive Streamlit dashboard for spam-message detection using
 K-Nearest Neighbors, Support Vector Machine and Naive Bayes classifiers.
==========================================================================================
Author : Generated for Message Intelligence Project
Run    : streamlit run app.py
==========================================================================================
"""

import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ==========================================================================================
#  PAGE CONFIGURATION
# ==========================================================================================
st.set_page_config(
    page_title="Message Intelligence System",
    page_icon="📨",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "Message_Intelligence_Dataset_5200_.csv"

# ==========================================================================================
#  GLOBAL STYLE  (custom CSS for a polished, modern, "product" feel)
# ==========================================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    :root{
        --mi-primary:#6C63FF;
        --mi-primary-dark:#4B3FE4;
        --mi-secondary:#00C9A7;
        --mi-danger:#FF5C7C;
        --mi-warning:#FF9671;
        --mi-bg-card:#13141F;
        --mi-border:rgba(255,255,255,0.08);
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #1b1c2c 0%, #0e0f1a 45%, #0a0a12 100%);
    }

    /* ---------- Hero header ---------- */
    .mi-hero{
        padding: 1.8rem 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(108,99,255,0.25) 0%, rgba(0,201,167,0.18) 100%);
        border: 1px solid var(--mi-border);
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 35px -10px rgba(108,99,255,0.35);
    }
    .mi-hero h1{
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #F5F5FF;
        letter-spacing: -0.5px;
    }
    .mi-hero p{
        color: #B7B8D9;
        font-size: 1.0rem;
        margin: 0;
    }
    .mi-badge{
        display:inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-right: 0.4rem;
        margin-top: 0.6rem;
        background: rgba(255,255,255,0.07);
        border: 1px solid var(--mi-border);
        color: #C9C9F2;
    }

    /* ---------- Generic card ---------- */
    .mi-card{
        background: var(--mi-bg-card);
        border: 1px solid var(--mi-border);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 18px -8px rgba(0,0,0,0.5);
        height: 100%;
    }
    .mi-card h4{
        margin-top:0;
        color:#F0F0FA;
        font-weight:700;
        font-size: 1.0rem;
    }
    .mi-card p, .mi-card span{
        color:#A7A8CC;
    }

    /* ---------- Metric tiles ---------- */
    .mi-metric{
        border-radius: 16px;
        padding: 1.0rem 1.2rem;
        border: 1px solid var(--mi-border);
        background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        text-align:left;
    }
    .mi-metric .label{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color:#9FA0C9;
    }
    .mi-metric .value{
        font-size: 1.7rem;
        font-weight: 800;
        color:#FFFFFF;
        margin-top: 0.15rem;
    }
    .mi-metric .delta{
        font-size: 0.8rem;
        color: var(--mi-secondary);
        font-weight: 600;
    }

    /* ---------- Verdict banner ---------- */
    .mi-verdict-spam{
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        background: linear-gradient(135deg, rgba(255,92,124,0.22), rgba(255,150,113,0.12));
        border: 1px solid rgba(255,92,124,0.4);
        text-align:center;
    }
    .mi-verdict-legit{
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        background: linear-gradient(135deg, rgba(0,201,167,0.22), rgba(108,99,255,0.12));
        border: 1px solid rgba(0,201,167,0.4);
        text-align:center;
    }
    .mi-verdict-spam h2, .mi-verdict-legit h2{
        margin: 0.2rem 0 0 0;
        font-size: 1.6rem;
        font-weight: 800;
        color: white;
    }
    .mi-verdict-spam .tag, .mi-verdict-legit .tag{
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.7);
    }

    /* ---------- Section title ---------- */
    .mi-section-title{
        font-size: 1.25rem;
        font-weight: 800;
        color: #F2F2FA;
        margin: 0.4rem 0 0.9rem 0;
        padding-left: 0.6rem;
        border-left: 4px solid var(--mi-primary);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"]{
        background: #0c0d16;
        border-right: 1px solid var(--mi-border);
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"]{
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"]{
        background-color: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        padding: 0.55rem 1.1rem;
        color:#B7B8D9;
        font-weight:600;
    }
    .stTabs [aria-selected="true"]{
        background-color: rgba(108,99,255,0.25) !important;
        color: white !important;
    }

    /* ---------- Misc ---------- */
    div[data-testid="stMetricValue"]{
        color:#F0F0FA;
    }
    hr{border-color: var(--mi-border);}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PALETTE = {
    "primary": "#6C63FF",
    "secondary": "#00C9A7",
    "warning": "#FF9671",
    "danger": "#FF5C7C",
    "muted": "#3D3F63",
}

PLOTLY_TEMPLATE = "plotly_dark"


def style_fig(fig, height=420):
    """Apply a consistent dark, transparent theme to every Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#D7D7EF"),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ==========================================================================================
#  DATA LOADING & FEATURE ENGINEERING
# ==========================================================================================
FEATURE_COLUMNS = [
    "message_length", "word_count", "num_urls", "num_digits", "num_special_chars",
    "spam_keyword_score", "legit_keyword_score", "sender_activity_score",
    "sender_account_age_days", "messages_sent_last_24h", "hour_of_day", "day_of_week",
]

SPAM_KEYWORDS = [
    "free", "win", "winner", "won", "prize", "cash", "urgent", "act fast", "act now",
    "limited offer", "limited time", "exclusive", "bonus", "claim", "reward", "refund",
    "click", "asap", "before it expires", "today only", "rich quick", "instant cashback",
    "earn money", "gift card", "gift cards", "suspended", "verify your account",
    "congratulations", "selected", "deal", "discount", "loan", "credit",
]

LEGIT_KEYWORDS = [
    "invoice", "report", "meeting", "schedule", "appointment", "confirm", "attached",
    "proposal", "review", "reminder", "payment received", "subscription", "quarterly",
    "regards", "thanks", "appreciate", "agenda", "timeline", "feedback", "discussion",
    "reschedule", "please find", "let me know",
]


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    """Load the raw CSV and apply the same cleaning used in the source notebook."""
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def extract_text_features(message: str) -> dict:
    """Derive the engineered numeric features directly from a raw message string,
    mirroring the columns present in the training dataset."""
    text = message or ""
    lower = text.lower()

    spam_score = sum(1 for kw in SPAM_KEYWORDS if kw in lower)
    legit_score = sum(1 for kw in LEGIT_KEYWORDS if kw in lower)

    return {
        "message_length": len(text),
        "word_count": len(text.split()),
        "num_urls": len(re.findall(r"https?://\S+|www\.\S+", lower)),
        "num_digits": sum(ch.isdigit() for ch in text),
        "num_special_chars": sum(1 for ch in text if not ch.isalnum() and not ch.isspace()),
        "spam_keyword_score": min(spam_score, 3),
        "legit_keyword_score": min(legit_score, 2),
    }


# ==========================================================================================
#  MODEL TRAINING  (cached so the expensive GridSearchCV work only runs once)
# ==========================================================================================
@st.cache_resource(show_spinner=False)
def train_pipeline(path: str):
    """Train KNN, SVM and Naive Bayes end-to-end, exactly following the workflow
    used in the original analysis notebook, and package everything the dashboard
    needs to render results."""

    df = load_dataset(path)

    X = df[FEATURE_COLUMNS]
    y = df["spam_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    results = {}

    # ---------------- KNN : sweep K = 1..20, keep the best ----------------
    k_values = list(range(1, 21))
    k_accuracies = []
    for k in k_values:
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train, y_train)
        k_accuracies.append(accuracy_score(y_test, m.predict(X_test)))

    best_k = k_values[int(np.argmax(k_accuracies))]
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)

    # distance metric comparison (extra diagnostic, fixed k=5 as in notebook)
    metric_scores = {}
    for metric in ["euclidean", "manhattan", "minkowski"]:
        m = KNeighborsClassifier(n_neighbors=5, metric=metric)
        m.fit(X_train, y_train)
        metric_scores[metric] = accuracy_score(y_test, m.predict(X_test))

    # ---------------- SVM : GridSearch over kernel / C / gamma ----------------
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid={"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
        cv=5, scoring="accuracy", n_jobs=-1,
    )
    svm_grid.fit(X_train, y_train)
    svm_model = svm_grid.best_estimator_
    svm_pred = svm_model.predict(X_test)

    # kernel comparison (extra diagnostic)
    kernel_scores = {}
    for kernel in ["linear", "rbf", "poly"]:
        kwargs = {"degree": 3} if kernel == "poly" else {}
        m = SVC(kernel=kernel, random_state=42, **kwargs)
        m.fit(X_train, y_train)
        kernel_scores[kernel] = accuracy_score(y_test, m.predict(X_test))

    # 2-D PCA projection purely for the support-vector visualization
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    svm_vis = SVC(kernel="linear")
    svm_vis.fit(X_train_pca, y_train)

    # ---------------- Naive Bayes : GridSearch over var_smoothing ----------------
    nb_grid = GridSearchCV(
        GaussianNB(),
        param_grid={"var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]},
        cv=5, scoring="accuracy",
    )
    nb_grid.fit(X_train, y_train)
    nb_model = nb_grid.best_estimator_
    nb_pred = nb_model.predict(X_test)
    nb_proba = nb_model.predict_proba(X_test)

    # ---------------- Consolidated metrics table ----------------
    model_preds = {"K-Nearest Neighbors": knn_pred, "Support Vector Machine": svm_pred, "Naive Bayes": nb_pred}
    metrics_rows = []
    for name, pred in model_preds.items():
        metrics_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
        })
    metrics_df = pd.DataFrame(metrics_rows)

    results.update(dict(
        df=df, scaler=scaler,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        knn_model=knn_model, best_k=best_k, k_values=k_values, k_accuracies=k_accuracies,
        metric_scores=metric_scores, knn_pred=knn_pred,
        svm_model=svm_model, svm_best_params=svm_grid.best_params_, kernel_scores=kernel_scores,
        svm_pred=svm_pred, pca=pca, X_train_pca=X_train_pca, svm_vis=svm_vis,
        nb_model=nb_model, nb_best_params=nb_grid.best_params_, nb_pred=nb_pred, nb_proba=nb_proba,
        metrics_df=metrics_df,
    ))
    return results


def predict_with_model(model_key: str, results: dict, raw_features: dict):
    """Scale a single hand-crafted feature row and run it through the chosen model."""
    feature_row = pd.DataFrame([raw_features])[FEATURE_COLUMNS]
    scaled_row = results["scaler"].transform(feature_row)

    model_map = {
        "K-Nearest Neighbors": results["knn_model"],
        "Support Vector Machine": results["svm_model"],
        "Naive Bayes": results["nb_model"],
    }
    model = model_map[model_key]
    pred = int(model.predict(scaled_row)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled_row)[0]
        proba_legit, proba_spam = float(proba[0]), float(proba[1])
    else:
        # SVC with probability=True already covers this, but keep a safe fallback
        score = model.decision_function(scaled_row)[0] if hasattr(model, "decision_function") else None
        proba_spam = 1.0 if pred == 1 else 0.0
        proba_legit = 1.0 - proba_spam

    return pred, proba_legit, proba_spam


# ==========================================================================================
#  LOAD DATA + TRAIN MODELS  (runs once, then served from cache)
# ==========================================================================================
with st.spinner("Loading dataset and training KNN / SVM / Naive Bayes models..."):
    R = train_pipeline(DATA_PATH)

df = R["df"]
metrics_df = R["metrics_df"]

# ==========================================================================================
#  SIDEBAR
# ==========================================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
            <div style="font-size:1.8rem;">📨</div>
            <div>
                <div style="font-weight:800;font-size:1.05rem;color:#F2F2FA;">Message Intelligence</div>
                <div style="font-size:0.75rem;color:#9FA0C9;">Spam Detection Console</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("##### 📦 Dataset Snapshot")
    c1, c2 = st.columns(2)
    c1.metric("Messages", f"{len(df):,}")
    c2.metric("Features", len(FEATURE_COLUMNS))
    c3, c4 = st.columns(2)
    c3.metric("Spam %", f"{(df['spam_label'].mean()*100):.1f}%")
    c4.metric("Legit %", f"{(100 - df['spam_label'].mean()*100):.1f}%")

    st.divider()
    st.markdown("##### 🤖 Best Models Found")
    st.caption(f"**KNN** → k = {R['best_k']}")
    st.caption(f"**SVM** → {R['svm_best_params']}")
    st.caption(f"**Naive Bayes** → var_smoothing = {R['nb_best_params']['var_smoothing']:.0e}")

    st.divider()
    st.markdown("##### ⚙️ Live Predictor Settings")
    sidebar_model_choice = st.selectbox(
        "Default model for the Predictor tab",
        ["Support Vector Machine", "Naive Bayes", "K-Nearest Neighbors"],
        index=0,
        help="You can still switch models inside the Predictor tab.",
    )

    st.divider()
    st.caption("Built with Streamlit · scikit-learn · Plotly")

# ==========================================================================================
#  HERO HEADER
# ==========================================================================================
st.markdown(
    f"""
    <div class="mi-hero">
        <h1>📨 Message Intelligence System</h1>
        <p>Probability-driven spam detection comparing K-Nearest Neighbors, Support Vector Machine
        and Naive Bayes classifiers on {len(df):,} real-world style messages.</p>
        <span class="mi-badge">scikit-learn</span>
        <span class="mi-badge">GridSearchCV Tuned</span>
        <span class="mi-badge">PCA Visualized</span>
        <span class="mi-badge">Live Inference</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================================
#  MAIN TABS
# ==========================================================================================
tab_overview, tab_explore, tab_models, tab_compare, tab_predict = st.tabs(
    ["🏠  Overview", "📊  Explore Data", "🤖  Model Training", "📈  Comparison", "🔮  Live Predictor"]
)

# ------------------------------------------------------------------------------------------
# TAB 1 : OVERVIEW
# ------------------------------------------------------------------------------------------
with tab_overview:
    best_row = metrics_df.loc[metrics_df["Accuracy"].idxmax()]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""<div class="mi-metric"><div class="label">Total Messages</div>
            <div class="value">{len(df):,}</div>
            <div class="delta">5,200 labeled samples</div></div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="mi-metric"><div class="label">Spam Ratio</div>
            <div class="value">{df['spam_label'].mean()*100:.1f}%</div>
            <div class="delta">{int(df['spam_label'].sum()):,} spam messages</div></div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="mi-metric"><div class="label">Best Model</div>
            <div class="value" style="font-size:1.25rem;">{best_row['Model']}</div>
            <div class="delta">{best_row['Accuracy']*100:.2f}% accuracy</div></div>""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""<div class="mi-metric"><div class="label">Engineered Features</div>
            <div class="value">{len(FEATURE_COLUMNS)}</div>
            <div class="delta">behavioral + textual signals</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown('<div class="mi-section-title">What this dashboard does</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="mi-card">
            <p>This console reproduces the full <b>Message Intelligence</b> analysis pipeline —
            from raw message features to tuned classifiers — in one interactive app.</p>
            <p>• <b>Explore Data</b> — distributions, correlations and behavioral patterns behind spam.<br>
            • <b>Model Training</b> — KNN K-sweep, SVM kernel search and Naive Bayes probability tuning.<br>
            • <b>Comparison</b> — side-by-side Accuracy / Precision / Recall / F1 with confusion matrices.<br>
            • <b>Live Predictor</b> — type a message, tune sender signals, and get an instant verdict.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="mi-section-title">Class balance</div>', unsafe_allow_html=True)
        donut = go.Figure(
            data=[go.Pie(
                labels=["Legitimate", "Spam"],
                values=[int((df["spam_label"] == 0).sum()), int((df["spam_label"] == 1).sum())],
                hole=0.62,
                marker=dict(colors=[PALETTE["secondary"], PALETTE["danger"]]),
                textinfo="percent",
                sort=False,
            )]
        )
        donut.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(style_fig(donut, height=300), use_container_width=True)

    st.write("")
    st.markdown('<div class="mi-section-title">Sample messages</div>', unsafe_allow_html=True)
    sample_view = df[["message_text", "spam_keyword_score", "legit_keyword_score", "hour_of_day", "spam_label"]].sample(
        8, random_state=7
    ).reset_index(drop=True)
    sample_view["spam_label"] = sample_view["spam_label"].map({0: "✅ Legitimate", 1: "🚨 Spam"})
    sample_view = sample_view.rename(columns={
        "message_text": "Message", "spam_keyword_score": "Spam KW", "legit_keyword_score": "Legit KW",
        "hour_of_day": "Hour", "spam_label": "Label",
    })
    st.dataframe(sample_view, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------------------
# TAB 2 : EXPLORE DATA
# ------------------------------------------------------------------------------------------
with tab_explore:
    st.markdown('<div class="mi-section-title">Feature distributions by class</div>', unsafe_allow_html=True)

    plot_df = df.copy()
    plot_df["Label"] = plot_df["spam_label"].map({0: "Legitimate", 1: "Spam"})

    feat_choice = st.selectbox(
        "Choose a feature to inspect",
        FEATURE_COLUMNS,
        index=FEATURE_COLUMNS.index("spam_keyword_score"),
    )

    c1, c2 = st.columns(2)
    with c1:
        hist = px.histogram(
            plot_df, x=feat_choice, color="Label", barmode="overlay", opacity=0.75,
            color_discrete_map={"Legitimate": PALETTE["secondary"], "Spam": PALETTE["danger"]},
        )
        hist.update_layout(title=f"Distribution of {feat_choice}")
        st.plotly_chart(style_fig(hist), use_container_width=True)
    with c2:
        box = px.box(
            plot_df, x="Label", y=feat_choice, color="Label",
            color_discrete_map={"Legitimate": PALETTE["secondary"], "Spam": PALETTE["danger"]},
        )
        box.update_layout(title=f"{feat_choice} spread by class", showlegend=False)
        st.plotly_chart(style_fig(box), use_container_width=True)

    st.write("")
    st.markdown('<div class="mi-section-title">Correlation between engineered features</div>', unsafe_allow_html=True)
    corr = df[FEATURE_COLUMNS + ["spam_label"]].corr()
    heat = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale=["#1b1c2c", "#6C63FF", "#00C9A7"],
    )
    heat.update_layout(title="Correlation Matrix")
    st.plotly_chart(style_fig(heat, height=520), use_container_width=True)

    st.write("")
    st.markdown('<div class="mi-section-title">When do spam messages arrive?</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        hour_rate = df.groupby("hour_of_day")["spam_label"].mean().reset_index()
        line = px.line(hour_rate, x="hour_of_day", y="spam_label", markers=True)
        line.update_traces(line_color=PALETTE["primary"])
        line.update_layout(title="Spam rate by hour of day", yaxis_title="Spam rate", xaxis_title="Hour")
        st.plotly_chart(style_fig(line), use_container_width=True)
    with c4:
        day_labels = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        day_rate = df.groupby("day_of_week")["spam_label"].mean().rename(index=day_labels).reset_index()
        bar = px.bar(day_rate, x="day_of_week", y="spam_label", color="spam_label",
                      color_continuous_scale=["#00C9A7", "#FF5C7C"])
        bar.update_layout(title="Spam rate by day of week", yaxis_title="Spam rate", xaxis_title="Day",
                           coloraxis_showscale=False)
        st.plotly_chart(style_fig(bar), use_container_width=True)


# ------------------------------------------------------------------------------------------
# TAB 3 : MODEL TRAINING
# ------------------------------------------------------------------------------------------
with tab_models:
    knn_tab, svm_tab, nb_tab = st.tabs(["📍 K-Nearest Neighbors", "🛡️ Support Vector Machine", "🎲 Naive Bayes"])

    # ---------------- KNN ----------------
    with knn_tab:
        c1, c2 = st.columns([1.4, 1])
        with c1:
            k_curve = go.Figure()
            k_curve.add_trace(go.Scatter(
                x=R["k_values"], y=R["k_accuracies"], mode="lines+markers",
                line=dict(color=PALETTE["primary"], width=3),
                marker=dict(size=7, color=PALETTE["secondary"]),
                name="Accuracy",
            ))
            k_curve.add_vline(x=R["best_k"], line_dash="dash", line_color=PALETTE["warning"])
            k_curve.update_layout(title="Accuracy across K values (1–20)", xaxis_title="K", yaxis_title="Accuracy")
            st.plotly_chart(style_fig(k_curve), use_container_width=True)
        with c2:
            st.markdown(
                f"""<div class="mi-card">
                <h4>Best configuration</h4>
                <p>Optimal neighbors <b>K = {R['best_k']}</b></p>
                <p>Test accuracy <b>{max(R['k_accuracies'])*100:.2f}%</b></p>
                <hr>
                <h4>Distance metric comparison (K=5)</h4>
                {''.join(f"<p>{m.title()} → <b>{s*100:.2f}%</b></p>" for m, s in R['metric_scores'].items())}
                </div>""",
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown('<div class="mi-section-title">KNN confusion matrix</div>', unsafe_allow_html=True)
        cm_knn = confusion_matrix(R["y_test"], R["knn_pred"])
        cm_fig = px.imshow(
            cm_knn, text_auto=True, color_continuous_scale=["#1b1c2c", "#6C63FF"],
            x=["Pred: Legit", "Pred: Spam"], y=["Actual: Legit", "Actual: Spam"],
        )
        cm_fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(cm_fig, height=380), use_container_width=True)

    # ---------------- SVM ----------------
    with svm_tab:
        c1, c2 = st.columns([1.4, 1])
        with c1:
            kernel_bar = px.bar(
                x=list(R["kernel_scores"].keys()), y=list(R["kernel_scores"].values()),
                color=list(R["kernel_scores"].keys()),
                color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"]],
                text=[f"{v*100:.2f}%" for v in R["kernel_scores"].values()],
            )
            kernel_bar.update_traces(textposition="outside")
            kernel_bar.update_layout(title="Accuracy by kernel (default params)", xaxis_title="Kernel",
                                      yaxis_title="Accuracy", showlegend=False, yaxis_range=[0, 1.05])
            st.plotly_chart(style_fig(kernel_bar), use_container_width=True)
        with c2:
            st.markdown(
                f"""<div class="mi-card">
                <h4>GridSearchCV result</h4>
                <p>Best params: <b>{R['svm_best_params']}</b></p>
                <p>Test accuracy <b>{accuracy_score(R['y_test'], R['svm_pred'])*100:.2f}%</b></p>
                <hr>
                <h4>Support vectors</h4>
                <p>Class 0 (Legit): <b>{R['svm_vis'].n_support_[0]}</b></p>
                <p>Class 1 (Spam): <b>{R['svm_vis'].n_support_[1]}</b></p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown('<div class="mi-section-title">Margin separation (PCA-projected, linear kernel)</div>', unsafe_allow_html=True)
        Xp, yv = R["X_train_pca"], R["y_train"].values
        sv = R["svm_vis"].support_vectors_
        pca_fig = go.Figure()
        for cls, color, name in [(0, PALETTE["secondary"], "Legitimate"), (1, PALETTE["danger"], "Spam")]:
            mask = yv == cls
            pca_fig.add_trace(go.Scatter(
                x=Xp[mask, 0], y=Xp[mask, 1], mode="markers", name=name,
                marker=dict(color=color, size=6, opacity=0.55),
            ))
        pca_fig.add_trace(go.Scatter(
            x=sv[:, 0], y=sv[:, 1], mode="markers", name="Support Vectors",
            marker=dict(color="rgba(0,0,0,0)", size=12, line=dict(color="white", width=1.5)),
        ))
        pca_fig.update_layout(title="SVM Support Vectors in 2-D PCA Space",
                               xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
        st.plotly_chart(style_fig(pca_fig, height=460), use_container_width=True)

    # ---------------- Naive Bayes ----------------
    with nb_tab:
        c1, c2 = st.columns([1.4, 1])
        with c1:
            sample_n = 12
            comp = pd.DataFrame({
                "Actual": R["y_test"].values[:sample_n],
                "Predicted": R["nb_pred"][:sample_n],
                "P(Legit)": R["nb_proba"][:sample_n, 0],
                "P(Spam)": R["nb_proba"][:sample_n, 1],
            }).round(4)
            prob_fig = go.Figure()
            prob_fig.add_trace(go.Bar(x=list(range(sample_n)), y=comp["P(Legit)"], name="P(Legit)",
                                       marker_color=PALETTE["secondary"]))
            prob_fig.add_trace(go.Bar(x=list(range(sample_n)), y=comp["P(Spam)"], name="P(Spam)",
                                       marker_color=PALETTE["danger"]))
            prob_fig.update_layout(barmode="stack", title="Bayes posterior probabilities (first 12 test messages)",
                                    xaxis_title="Test sample index", yaxis_title="Probability")
            st.plotly_chart(style_fig(prob_fig), use_container_width=True)
        with c2:
            st.markdown(
                f"""<div class="mi-card">
                <h4>GridSearchCV result</h4>
                <p>Best <code>var_smoothing</code>: <b>{R['nb_best_params']['var_smoothing']:.0e}</b></p>
                <p>Test accuracy <b>{accuracy_score(R['y_test'], R['nb_pred'])*100:.2f}%</b></p>
                <hr>
                <h4>Why it works</h4>
                <p>Naive Bayes applies Bayes' theorem assuming feature independence — fast,
                probability-native, and well suited to short, keyword-driven messages.</p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown('<div class="mi-section-title">Sample probability table</div>', unsafe_allow_html=True)
        comp_display = comp.copy()
        comp_display["Actual"] = comp_display["Actual"].map({0: "Legit", 1: "Spam"})
        comp_display["Predicted"] = comp_display["Predicted"].map({0: "Legit", 1: "Spam"})
        st.dataframe(comp_display, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------------------
# TAB 4 : COMPARISON
# ------------------------------------------------------------------------------------------
with tab_compare:
    st.markdown('<div class="mi-section-title">Accuracy · Precision · Recall · F1 Score</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    icons = {"K-Nearest Neighbors": "📍", "Support Vector Machine": "🛡️", "Naive Bayes": "🎲"}
    for col, (_, row) in zip(cols, metrics_df.iterrows()):
        with col:
            st.markdown(
                f"""<div class="mi-card">
                <h4>{icons.get(row['Model'],'🤖')} {row['Model']}</h4>
                <p style="font-size:1.7rem;font-weight:800;color:#fff;margin:0;">{row['Accuracy']*100:.2f}%</p>
                <span>Accuracy</span>
                <hr>
                <p>Precision <b>{row['Precision']*100:.2f}%</b></p>
                <p>Recall <b>{row['Recall']*100:.2f}%</b></p>
                <p>F1 Score <b>{row['F1 Score']*100:.2f}%</b></p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        grouped_bar = px.bar(
            melted, x="Model", y="Score", color="Metric", barmode="group",
            color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"], PALETTE["danger"]],
            text=melted["Score"].apply(lambda v: f"{v*100:.1f}%"),
        )
        grouped_bar.update_traces(textposition="outside")
        grouped_bar.update_layout(title="Model comparison across all metrics", yaxis_range=[0, 1.15])
        st.plotly_chart(style_fig(grouped_bar), use_container_width=True)
    with c2:
        best_precision = metrics_df.loc[metrics_df["Precision"].idxmax()]
        best_recall = metrics_df.loc[metrics_df["Recall"].idxmax()]
        best_f1 = metrics_df.loc[metrics_df["F1 Score"].idxmax()]
        st.markdown(
            f"""<div class="mi-card">
            <h4>🏆 Category leaders</h4>
            <p>Highest Precision → <b>{best_precision['Model']}</b> ({best_precision['Precision']*100:.2f}%)</p>
            <p>Highest Recall → <b>{best_recall['Model']}</b> ({best_recall['Recall']*100:.2f}%)</p>
            <p>Highest F1 Score → <b>{best_f1['Model']}</b> ({best_f1['F1 Score']*100:.2f}%)</p>
            <hr>
            <p><b>Precision</b> minimizes false alarms on legitimate messages.<br>
            <b>Recall</b> maximizes how much real spam gets caught.<br>
            Choose based on which mistake costs you more.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown('<div class="mi-section-title">Confusion matrices, side by side</div>', unsafe_allow_html=True)
    cm_cols = st.columns(3)
    cm_data = {
        "K-Nearest Neighbors": R["knn_pred"],
        "Support Vector Machine": R["svm_pred"],
        "Naive Bayes": R["nb_pred"],
    }
    for col, (name, pred) in zip(cm_cols, cm_data.items()):
        with col:
            cm = confusion_matrix(R["y_test"], pred)
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale=["#1b1c2c", "#6C63FF"],
                x=["Pred Legit", "Pred Spam"], y=["Actual Legit", "Actual Spam"],
            )
            fig.update_layout(title=name, coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, height=340), use_container_width=True)

    st.write("")
    st.markdown('<div class="mi-section-title">Full metrics table</div>', unsafe_allow_html=True)
    styled = metrics_df.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        styled[c] = styled[c].apply(lambda v: f"{v*100:.2f}%")
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------------------
# TAB 5 : LIVE PREDICTOR
# ------------------------------------------------------------------------------------------
with tab_predict:
    st.markdown('<div class="mi-section-title">Test a message in real time</div>', unsafe_allow_html=True)

    st.session_state.setdefault(
        "message_box", "URGENT: Claim your reward now before it expires! https://bit.ly/4187"
    )

    left, right = st.columns([1.15, 1])

    with left:
        message_input = st.text_area(
            "Message text",
            key="message_box",
            height=110,
            help="Type or paste any message — text features are derived automatically.",
        )

        model_choice = st.radio(
            "Classifier",
            ["Support Vector Machine", "Naive Bayes", "K-Nearest Neighbors"],
            index=["Support Vector Machine", "Naive Bayes", "K-Nearest Neighbors"].index(sidebar_model_choice),
            horizontal=True,
        )

        st.markdown("##### Sender & timing signals")
        s1, s2 = st.columns(2)
        with s1:
            sender_activity_score = st.slider("Sender activity score", 0.0, 40.0, 22.0, 0.5)
            sender_account_age_days = st.slider("Account age (days)", 1.0, 1500.0, 300.0, 1.0)
        with s2:
            messages_sent_last_24h = st.slider("Messages sent in last 24h", 0.0, 38.0, 4.0, 1.0)
            hour_of_day = st.slider("Hour of day", 0, 23, 14)

        day_of_week = st.select_slider(
            "Day of week",
            options=[0, 1, 2, 3, 4, 5, 6],
            value=2,
            format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
        )

        run_predict = st.button("🔍 Classify Message", type="primary", use_container_width=True)

    derived = extract_text_features(message_input)
    raw_features = {
        **derived,
        "sender_activity_score": sender_activity_score,
        "sender_account_age_days": sender_account_age_days,
        "messages_sent_last_24h": messages_sent_last_24h,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
    }

    with right:
        st.markdown("##### Auto-derived text features")
        f1, f2, f3 = st.columns(3)
        f1.metric("Length", derived["message_length"])
        f2.metric("Words", derived["word_count"])
        f3.metric("URLs", derived["num_urls"])
        f4, f5, f6 = st.columns(3)
        f4.metric("Digits", derived["num_digits"])
        f5.metric("Spam KW", derived["spam_keyword_score"])
        f6.metric("Legit KW", derived["legit_keyword_score"])

        st.write("")
        if run_predict or message_input.strip():
            pred, proba_legit, proba_spam = predict_with_model(model_choice, R, raw_features)
            time.sleep(0.05)

            if pred == 1:
                st.markdown(
                    f"""<div class="mi-verdict-spam">
                    <div class="tag">⚠️ Verdict</div>
                    <h2>🚨 Likely SPAM</h2>
                    <div class="tag">Confidence {proba_spam*100:.1f}%</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="mi-verdict-legit">
                    <div class="tag">✅ Verdict</div>
                    <h2>Legitimate Message</h2>
                    <div class="tag">Confidence {proba_legit*100:.1f}%</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.write("")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba_spam * 100,
                number={"suffix": "%"},
                title={"text": "Probability of Spam"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PALETTE["danger"] if pred == 1 else PALETTE["secondary"]},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(0,201,167,0.25)"},
                        {"range": [40, 70], "color": "rgba(255,150,113,0.25)"},
                        {"range": [70, 100], "color": "rgba(255,92,124,0.25)"},
                    ],
                },
            ))
            st.plotly_chart(style_fig(gauge, height=260), use_container_width=True)
            st.caption(f"Model used: **{model_choice}** · Engine: scikit-learn pipeline trained on {len(df):,} messages")
        else:
            st.info("Type a message on the left and press **Classify Message** to see the verdict.")

    st.write("")
    st.markdown('<div class="mi-section-title">Try a few quick examples</div>', unsafe_allow_html=True)
    examples = {
        "💸 Prize scam": "Congratulations! You've been selected for a cash prize, claim now before it expires!",
        "📎 Work email": "Hi, can we reschedule our meeting to 10:30 AM tomorrow? Please confirm.",
        "🏦 Loan spam": "Get instant cashback and a free gift card, act fast, limited offer today only!",
    }
    def _load_example(example_text: str):
        # Runs BEFORE the script reruns/redraws widgets, so it's safe to
        # write into session_state here (unlike doing it after the widget
        # has already been instantiated in the same run).
        st.session_state["message_box"] = example_text

    ex_cols = st.columns(len(examples))
    for col, (label, text) in zip(ex_cols, examples.items()):
        with col:
            st.button(
                label, use_container_width=True, key=f"ex_{label}",
                on_click=_load_example, args=(text,),
            )


# ==========================================================================================
#  FOOTER
# ==========================================================================================
st.write("")
st.markdown(
    """
    <div style="text-align:center;padding:1.2rem 0 0.4rem 0;color:#6F70A0;font-size:0.82rem;">
        Message Intelligence System · KNN · SVM · Naive Bayes · Built with Streamlit & Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
