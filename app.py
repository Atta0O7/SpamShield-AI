
from pathlib import Path
from datetime import datetime
import re

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# 1. APP CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_detector_model.joblib"
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"


# =========================================================
# 2. LOAD MODEL AND DATA
# =========================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["label", "message"])

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        on_bad_lines="skip",
    )

    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["message"] = df["message"].astype(str)
    df = df[df["label"].isin(["ham", "spam"])]
    df = df.drop_duplicates().reset_index(drop=True)
    df["message_length"] = df["message"].str.len()

    return df


model = load_model()
data = load_data()

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_message" not in st.session_state:
    st.session_state.last_message = ""


# =========================================================
# 3. PREMIUM GLASSMORPHISM DESIGN
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 8% 5%, rgba(73,111,255,.22), transparent 38%),
        radial-gradient(ellipse at 92% 10%, rgba(218,91,255,.15), transparent 35%),
        linear-gradient(135deg, #080d1b, #101a31 55%, #17132c);
    color: #edf3ff;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(8,13,27,.35);
}

.block-container {
    max-width: 1450px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f5f7ff !important;
    letter-spacing: -.4px;
}

p, label, .stMarkdown {
    color: #d7e1f7;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(40,60,100,.48),
                rgba(38,31,75,.52));
    border: 1px solid rgba(160,180,255,.25);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 28px rgba(0,0,0,.16);
}

[data-testid="stMetricLabel"] {
    color: #c6d5f3 !important;
}

[data-testid="stMetricValue"] {
    color: #8bd8ff !important;
    font-weight: 800 !important;
}

.stTextArea textarea,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] {
    background: rgba(16,27,51,.82);
    color: white;
    border-radius: 12px;
    border-color: rgba(140,165,230,.42);
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(100deg,#579cff,#907aff,#d982e8);
    color: white;
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 12px;
    font-weight: 700;
    min-height: 44px;
    transition: all .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    color: white;
    border-color: white;
    box-shadow: 0 0 20px rgba(139,130,255,.3);
    transform: translateY(-1px);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(160,180,255,.2);
}

.stTabs [aria-selected="true"] {
    color: #9bdcff !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a31, #17132c);
    border-right: 1px solid rgba(160,180,255,.18);
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

hr {
    border-color: rgba(160,180,255,.2);
}

footer {
    visibility: hidden;
}

.small-muted {
    color: #9cafd2;
    font-size: 13px;
}

.hero {
    padding: 26px;
    border: 1px solid rgba(160,180,255,.24);
    border-radius: 24px;
    background: linear-gradient(120deg,
        rgba(50,75,125,.38), rgba(72,42,100,.27));
    margin-bottom: 22px;
}

.hero-label {
    font-size: 12px;
    letter-spacing: 3px;
    color: #9bdcff;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. PREDICTION AND EXPLANATION HELPERS
# =========================================================
def normalize_label(value):
    """Convert common classifier outputs to ham/spam labels."""
    label = str(value).strip().lower()

    if label in ("spam", "1", "true"):
        return "spam"

    if label in ("ham", "not spam", "not_spam", "0", "false"):
        return "ham"

    return label


def classify_message(message):
    """Predict a message and return available class probabilities."""
    prediction_raw = model.predict([message])[0]
    prediction = normalize_label(prediction_raw)

    probabilities = {}

    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba([message])[0]
        classes = list(model.classes_)

        for label, probability in zip(classes, raw_probs):
            probabilities[normalize_label(label)] = float(probability)

    spam_probability = probabilities.get("spam")
    ham_probability = probabilities.get("ham")

    if prediction in probabilities:
        confidence = probabilities[prediction]
    else:
        confidence = None

    return {
        "prediction": prediction,
        "confidence": confidence,
        "spam_probability": spam_probability,
        "ham_probability": ham_probability,
    }


# These patterns are explanatory clues, not proof that a message is spam.
SUSPICIOUS_PATTERNS = {
    "Prize or reward language": [
        r"\bwon\b", r"\bwinner\b", r"\bprize\b", r"\breward\b",
        r"\bfree gift\b", r"\bclaim your\b",
    ],
    "Urgency or pressure": [
        r"\bact now\b", r"\burgent\b", r"\bimmediately\b",
        r"\bexpires today\b", r"\blimited time\b",
    ],
    "Money or financial claims": [
        r"\bcash\b", r"\bbonus\b", r"\bloan\b",
        r"\bprofit\b", r"\b\d+\s*(?:rs|inr|rupees|usd|dollars)\b",
    ],
    "Suspicious link": [
        r"https?://", r"\bwww\.", r"\bbit\.ly\b",
        r"\btinyurl\b",
    ],
    "Requests for sensitive details": [
        r"\botp\b", r"\bpassword\b", r"\bpin number\b",
        r"\bverify your account\b", r"\bbank details\b",
    ],
    "Unsolicited contact or offer": [
        r"\bcall now\b", r"\bclick here\b", r"\btext stop\b",
        r"\bsubscribe\b",
    ],
}


def explain_message(message, result):
    """Generate transparent, rule-based clues alongside model prediction."""
    lowered = message.lower()
    matched = []

    for category, patterns in SUSPICIOUS_PATTERNS.items():
        hits = []

        for pattern in patterns:
            for match in re.finditer(pattern, lowered):
                hits.append(match.group(0))

        if hits:
            matched.append({
                "category": category,
                "terms": sorted(set(hits)),
            })

    spam_probability = result["spam_probability"]

    if spam_probability is None:
        risk = "Model probability unavailable"
    elif spam_probability >= 0.80:
        risk = "High"
    elif spam_probability >= 0.50:
        risk = "Medium"
    else:
        risk = "Low"

    return risk, matched


def add_to_history(message, result, risk):
    """Record one prediction in the current Streamlit session."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
        "prediction": result["prediction"].upper(),
        "risk": risk,
        "spam_probability": (
            round(result["spam_probability"] * 100, 2)
            if result["spam_probability"] is not None else None
        ),
        "confidence": (
            round(result["confidence"] * 100, 2)
            if result["confidence"] is not None else None
        ),
    }

    st.session_state.history.insert(0, entry)


def show_probability_chart(result):
    spam_prob = result["spam_probability"]
    ham_prob = result["ham_probability"]

    if spam_prob is None or ham_prob is None:
        st.info(
            "Saved model probability scores provide nahi karta. "
            "Prediction label phir bhi available ho sakta hai."
        )
        return

    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#111a30")
    ax.set_facecolor("#111a30")

    values = [ham_prob * 100, spam_prob * 100]
    labels = ["Legitimate (Ham)", "Spam"]

    bars = ax.barh(
        labels,
        values,
        color=["#61d9b5", "#ff719a"],
        height=.48,
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Model probability (%)", color="#d7e1f7")
    ax.tick_params(colors="#d7e1f7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#53617e")
    ax.spines["left"].set_color("#53617e")

    for bar, value in zip(bars, values):
        ax.text(
            min(value + 1, 93),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            color="white",
            va="center",
            fontweight="bold",
        )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =========================================================
# 5. HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-label">INTELLIGENT MESSAGE SECURITY</div>
    <h1 style="font-size:42px;margin:8px 0;">🛡️ SpamShield AI</h1>
    <p style="font-size:17px;margin-bottom:8px;">
        Detect suspicious SMS messages with machine learning.
    </p>
    <div class="small-muted">
        Text classification • Risk insights • Bulk analysis
    </div>
</div>
""", unsafe_allow_html=True)


if model is None:
    st.error(
        "Model file nahi mila. Check this path: "
        "models/spam_detector_model.joblib"
    )
    st.stop()


# =========================================================
# 6. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🛡️ SpamShield AI")
    st.caption("Your SMS analysis workspace")
    st.divider()

    st.markdown("### 📌 Quick Stats")

    st.metric("Predictions this session", len(st.session_state.history))

    if not data.empty:
        st.metric("Dataset messages", f"{len(data):,}")
    else:
        st.metric("Dataset messages", "Unavailable")

    st.divider()
    st.markdown("### 🧭 Navigation")
    st.caption("Choose a section from the tabs on the main screen.")

    st.divider()
    st.caption("Educational ML project • Python + Streamlit")


# =========================================================
# 7. NAVIGATION TABS
# =========================================================
home_tab, bulk_tab, history_tab, analytics_tab, about_tab = st.tabs([
    "🏠 Detector",
    "📂 Bulk Analysis",
    "🕘 History",
    "📊 Analytics",
    "ℹ️ About",
])


# =========================================================
# 8. HOME: SINGLE MESSAGE DETECTOR
# =========================================================
with home_tab:
    st.subheader("🔍 Analyze a Message")
    st.write("Enter an SMS to get the model's prediction and risk clues.")

    message = st.text_area(
        "Message text",
        placeholder=(
            "Example: Congratulations! You won a free prize. "
            "Call now to claim it!"
        ),
        height=145,
        key="single_message",
    )

    sample1, sample2, sample3 = st.columns(3)

    with sample1:
        if st.button("🚨 Try spam example", use_container_width=True):
            st.session_state.single_message = (
                "Congratulations! You won a free prize. "
                "Call now to claim it!"
            )
            st.rerun()

    with sample2:
        if st.button("💬 Try normal example", use_container_width=True):
            st.session_state.single_message = (
                "Hi, I will reach college in 10 minutes. "
                "Please wait for me."
            )
            st.rerun()

    with sample3:
        if st.button("🧹 Clear previous result", use_container_width=True):
            st.session_state.last_result = None
            st.session_state.last_message = ""
            st.rerun()

    if st.button("🚀 Analyze Message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Pehle message enter karo.")
        else:
            try:
                result = classify_message(message.strip())
                risk, clues = explain_message(message.strip(), result)

                st.session_state.last_result = result
                st.session_state.last_message = message.strip()
                st.session_state.last_risk = risk
                st.session_state.last_clues = clues

                add_to_history(message.strip(), result, risk)

            except Exception as error:
                st.error(f"Prediction error: {error}")

    if st.session_state.last_result is not None:
        result = st.session_state.last_result
        risk = st.session_state.get("last_risk", "Unknown")
        clues = st.session_state.get("last_clues", [])

        st.divider()
        st.subheader("🎯 Prediction Result")

        left, right = st.columns([1.1, 1])

        with left:
            if result["prediction"] == "spam":
                st.error("🚨 SPAM DETECTED")
            elif result["prediction"] == "ham":
                st.success("✅ LIKELY LEGITIMATE")
            else:
                st.warning(f"Prediction: {result['prediction']}")

            a, b = st.columns(2)
            a.metric(
                "Predicted class",
                result["prediction"].upper(),
            )
            b.metric(
                "Model confidence",
                (
                    f"{result['confidence'] * 100:.2f}%"
                    if result["confidence"] is not None
                    else "Unavailable"
                ),
            )

        with right:
            st.markdown("#### 🚦 Risk Assessment")
            if risk == "High":
                st.error("High estimated spam probability")
            elif risk == "Medium":
                st.warning("Medium estimated spam probability")
            elif risk == "Low":
                st.success("Low estimated spam probability")
            else:
                st.info(risk)

            st.caption(
                "Risk levels are based on the model's spam probability, "
                "when available."
            )

        st.subheader("📈 Class Probabilities")
        show_probability_chart(result)

        st.subheader("🧠 Why might this message be suspicious?")

        if clues:
            for clue in clues:
                terms = ", ".join(clue["terms"])
                st.markdown(f"- **{clue['category']}** — matched: `{terms}`")
        else:
            st.info(
                "No configured suspicious phrase was found. "
                "This does not guarantee that the message is safe."
            )

        st.caption(
            "Explanation is rule-based and highlights selected text patterns. "
            "It does not show the exact internal reasoning of the ML model."
        )

        with st.expander("View analyzed message"):
            st.write(st.session_state.last_message)

        st.warning(
            "Always verify unexpected links, OTP requests, payment requests, "
            "and unknown senders. A model can make mistakes."
        )


# =========================================================
# 9. BULK CSV ANALYSIS
# =========================================================
with bulk_tab:
    st.subheader("📂 Bulk SMS Analyzer")
    st.write(
        "Upload a CSV containing multiple SMS messages. "
        "The app will predict each message and prepare a downloadable report."
    )

    st.markdown("#### Expected CSV format")
    st.code(
        "message\n"
        "Congratulations! You won a prize\n"
        "Hi, I will reach in 10 minutes",
        language="csv",
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="bulk_csv",
        help="The CSV should contain a column with message text.",
    )

    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)

            if bulk_df.empty:
                st.warning("Uploaded CSV is empty.")
            else:
                st.write("#### Preview")
                st.dataframe(
                    bulk_df.head(10),
                    use_container_width=True,
                )

                columns = list(bulk_df.columns)

                suggested = next(
                    (
                        column for column in columns
                        if str(column).strip().lower()
                        in ["message", "sms", "text", "message_text"]
                    ),
                    columns[0],
                )

                message_column = st.selectbox(
                    "Which column contains SMS messages?",
                    options=columns,
                    index=columns.index(suggested),
                )

                max_rows = st.number_input(
                    "Maximum rows to analyze",
                    min_value=1,
                    max_value=10000,
                    value=min(len(bulk_df), 1000),
                    step=100,
                )

                if st.button(
                    "🔎 Analyze CSV",
                    type="primary",
                    use_container_width=True,
                ):
                    working = bulk_df.head(int(max_rows)).copy()

                    valid_messages = working[message_column].fillna("").astype(str)
                    output_rows = []

                    progress = st.progress(0)
                    status = st.empty()

                    for index, text_value in enumerate(valid_messages):
                        cleaned = text_value.strip()

                        if not cleaned:
                            output_rows.append({
                                "message": text_value,
                                "prediction": "EMPTY",
                                "spam_probability": None,
                                "risk": "Unknown",
                                "suspicious_clues": "",
                            })
                        else:
                            result = classify_message(cleaned)
                            risk, clues = explain_message(cleaned, result)

                            output_rows.append({
                                "message": text_value,
                                "prediction": result["prediction"].upper(),
                                "spam_probability": (
                                    round(result["spam_probability"] * 100, 2)
                                    if result["spam_probability"] is not None
                                    else None
                                ),
                                "risk": risk,
                                "suspicious_clues": "; ".join(
                                    clue["category"] + ": "
                                    + ", ".join(clue["terms"])
                                    for clue in clues
                                ),
                            })

                            add_to_history(cleaned, result, risk)

                        progress.progress((index + 1) / len(valid_messages))

                    output_df = pd.DataFrame(output_rows)

                    # Keep the output and original columns together.
                    output_df = pd.concat(
                        [
                            working.reset_index(drop=True),
                            output_df.drop(columns=["message"]).reset_index(drop=True),
                        ],
                        axis=1,
                    )

                    st.session_state.bulk_results = output_df
                    status.success(
                        f"Analysis complete: {len(output_df):,} rows processed."
                    )

        except Exception as error:
            st.error(
                f"Could not read or analyze this CSV: {error}. "
                "Please check the file format and message column."
            )

    if "bulk_results" in st.session_state:
        results_df = st.session_state.bulk_results

        st.divider()
        st.subheader("📋 Bulk Results")

        spam_total = int((results_df["prediction"] == "SPAM").sum())
        ham_total = int((results_df["prediction"] == "HAM").sum())
        empty_total = int((results_df["prediction"] == "EMPTY").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Spam found", spam_total)
        c2.metric("Legitimate found", ham_total)
        c3.metric("Empty messages", empty_total)

        st.dataframe(results_df, use_container_width=True, height=350)

        st.download_button(
            "⬇️ Download prediction report",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="spamshield_bulk_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# 10. SESSION PREDICTION HISTORY
# =========================================================
with history_tab:
    st.subheader("🕘 Prediction History")
    st.write(
        "Single-message aur bulk predictions current session mein yahan dikhengi."
    )

    history = st.session_state.history

    if not history:
        st.info("Abhi history empty hai. Pehle koi message analyze karo.")
    else:
        history_df = pd.DataFrame(history)

        total = len(history_df)
        spam_count = int((history_df["prediction"] == "SPAM").sum())
        ham_count = int((history_df["prediction"] == "HAM").sum())

        h1, h2, h3 = st.columns(3)
        h1.metric("Total predictions", total)
        h2.metric("Spam predictions", spam_count)
        h3.metric("Ham predictions", ham_count)

        search_text = st.text_input(
            "Search history",
            placeholder="Search message text...",
        )

        filtered_history = history_df.copy()

        if search_text.strip():
            filtered_history = filtered_history[
                filtered_history["message"].str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        st.dataframe(
            filtered_history,
            use_container_width=True,
            height=400,
        )

        st.download_button(
            "⬇️ Export history as CSV",
            data=filtered_history.to_csv(index=False).encode("utf-8"),
            file_name="spamshield_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("🗑️ Clear prediction history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.caption(
        "History is stored in Streamlit session state, not a permanent database. "
        "It may disappear after the session resets."
    )


# =========================================================
# 11. DATASET ANALYTICS
# =========================================================
with analytics_tab:
    st.subheader("📊 Dataset Analytics")

    if data.empty:
        st.warning(
            "Dataset unavailable. Check data/SMSSpamCollection."
        )
    else:
        spam_count = int((data["label"] == "spam").sum())
        ham_count = int((data["label"] == "ham").sum())

        a1, a2, a3 = st.columns(3)
        a1.metric("Total messages", f"{len(data):,}")
        a2.metric("Spam messages", f"{spam_count:,}")
        a3.metric("Ham messages", f"{ham_count:,}")

        left, right = st.columns(2)

        with left:
            st.markdown("#### Message distribution")

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#111a30")
            ax.set_facecolor("#111a30")

            ax.pie(
                [ham_count, spam_count],
                labels=["Ham", "Spam"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#61d9b5", "#ff719a"],
                textprops={"color": "white"},
                wedgeprops={"width": .42, "edgecolor": "#111a30"},
            )

            ax.set_title("Ham vs Spam", color="white")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with right:
            st.markdown("#### Average message length")

            averages = data.groupby("label")["message_length"].mean()
            averages = averages.reindex(["ham", "spam"])

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#111a30")
            ax.set_facecolor("#111a30")

            bars = ax.bar(
                ["Ham", "Spam"],
                averages.values,
                color=["#61d9b5", "#ff719a"],
                width=.55,
            )

            ax.set_ylabel("Characters", color="#d7e1f7")
            ax.tick_params(colors="#d7e1f7")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{bar.get_height():.0f}",
                    ha="center",
                    color="white",
                )

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.subheader("🔎 Search dataset")

        search = st.text_input(
            "Search dataset messages",
            placeholder="Enter a word or phrase...",
        )

        filtered = data.copy()

        if search.strip():
            filtered = filtered[
                filtered["message"].str.contains(
                    search.strip(),
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        st.dataframe(
            filtered[["label", "message", "message_length"]],
            use_container_width=True,
            height=350,
        )

        st.download_button(
            "⬇️ Download dataset CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="spamshield_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# 12. ABOUT
# =========================================================
with about_tab:
    st.subheader("ℹ️ About SpamShield AI")

    st.markdown("""
    **SpamShield AI** is an SMS spam classification project built using
    Python, Scikit-learn, Joblib, Pandas, Matplotlib, and Streamlit.

    ### Main features
    - Single SMS classification
    - Model probability display, when supported
    - Rule-based suspicious phrase insights
    - Session-based prediction history
    - Bulk CSV prediction and export
    - Dataset analytics

    ### How it works
    1. The user enters a message or uploads a CSV.
    2. The saved machine learning model predicts the message class.
    3. Available probabilities and rule-based clues are displayed.
    4. Results can be downloaded for further analysis.

    ### Important limitations
    - Predictions can be wrong.
    - Suspicious phrases are clues, not proof of fraud.
    - History is session-based and not permanently stored.
    - Model probability is not a guarantee of correctness.

    Never share OTPs, passwords, bank PINs, or other sensitive information.
    """)

    st.divider()
    st.caption("SpamShield AI • Machine Learning Portfolio Project")
