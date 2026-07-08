import re
import io
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Student AI/ML Project", page_icon="🎓", layout="wide")


RAW_CSV = """Student_ID,Name,Age,Gender,Math,Science,English,Attendance,Grade
101,John,18,Male,85,78,90,95,A
102,Emma,17,Female,92,88,84,98,A
103,Michael,18,Male,76,81,79,90,B
104,Sophia,17,Female,89,93,91,97,A
105,David,19,Male,65,70,68,85,C
106,Olivia,18,Female,95,96,94,99,A
107,James,17,Male,72,75,80,88,B
108,Ava,18,Female,88,85,87,92,B
109,William,19,Male,60,66,64,82,C
110,Isabella,17,Female,91,89,93,96,A
"""

FEEDBACK = {
    101: "The classes were good and I enjoyed the science lab.",
    102: "Excellent teaching, I loved every subject this term.",
    103: "It was okay, some topics were a bit confusing.",
    104: "Great experience, the teachers were very helpful.",
    105: "I struggled a lot, the pace felt too fast for me.",
    106: "Amazing school year, everything was fun and clear.",
    107: "Average experience, could be more engaging.",
    108: "Good overall but the homework load was heavy.",
    109: "I found the exams difficult and quite stressful.",
    110: "Wonderful teachers, I feel confident and satisfied.",
}

# Simple word lists used for sentiment analysis (NLP)
POSITIVE_WORDS = {"good", "great", "excellent", "amazing", "loved", "helpful",
                   "fun", "clear", "wonderful", "confident", "satisfied", "enjoyed"}
NEGATIVE_WORDS = {"struggled", "confusing", "difficult", "stressful", "heavy",
                   "bad", "poor", "hard"}
STOPWORDS = {"a", "an", "the", "is", "was", "were", "i", "it", "this", "of",
             "and", "to", "for", "my", "some", "very", "felt", "found"}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return clean_text(text).split()


def get_sentiment(text):
    words = tokenize(text)
    pos = sum(w in POSITIVE_WORDS for w in words)
    neg = sum(w in NEGATIVE_WORDS for w in words)
    if pos > neg:
        return "Positive "
    elif neg > pos:
        return "Negative "
    return "Neutral "


def extract_text(resp):
    """
    Claude Sonnet 5 uses adaptive thinking, so resp.content can include a
    ThinkingBlock before the actual text reply. This pulls out only the
    real text block instead of blindly assuming content[0] is text.
    Returns None (not "") if no text block was found, so callers can
    tell "empty on purpose" apart from "nothing came back at all".
    """
    for block in resp.content:
        if block.type == "text" and block.text.strip():
            return block.text
    return None


# DATABASE (SQLite) - loads your CSV data once, then remembers changes

@st.cache_resource
def get_connection():
    # Use an in-memory DB. Streamlit Community Cloud's app directory
    # (/mount/src/...) is read-only, so writing a "students.db" file
    # there fails with an OperationalError. We don't need the data to
    # survive an app restart anyway (it's rebuilt from RAW_CSV below),
    # and st.cache_resource keeps this exact connection object alive
    # across reruns within the same session, so in-memory data persists
    # for as long as the app process is running.
    return sqlite3.connect(":memory:", check_same_thread=False)


def init_db():
    conn = get_connection()
    df = pd.read_csv(io.StringIO(RAW_CSV))
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='students'"
    ).fetchone()
    if not exists:
        df["Feedback"] = df["Student_ID"].map(FEEDBACK)
        df.to_sql("students", conn, index=False)


def get_students():
    return pd.read_sql("SELECT * FROM students", get_connection())


def add_student(row):
    conn = get_connection()
    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    conn.execute(f"INSERT INTO students ({cols}) VALUES ({placeholders})", tuple(row.values()))
    conn.commit()


def delete_student(student_id):
    conn = get_connection()
    conn.execute("DELETE FROM students WHERE Student_ID = ?", (student_id,))
    conn.commit()


init_db()


# SIDEBAR

st.sidebar.title(" Menu")
page = st.sidebar.radio("Go to:", [
    " Home",
    " Database",
    " EDA",
    " Classification (predict Grade)",
    " Regression (predict Science score)",
    " Clustering (group students)",
    " NLP - Feedback",
    " RAG - Q&A",
    " GenAI Chatbot",

])

st.sidebar.markdown("---")

# Prefer a key stored in Streamlit secrets (safe, never shown on screen).
# Falls back to the visible text input only if no secret is configured.
# To use secrets: create .streamlit/secrets.toml with:
#   ANTHROPIC_API_KEY = "sk-ant-..."
_secret_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""

if _secret_key:
    api_key = _secret_key
    st.sidebar.success(" Using API key from Streamlit secrets.")
else:
    api_key = st.sidebar.text_input(" Claude API Key (optional)", type="password",
                                     help="Leave blank to still use the app with a simple fallback.")
    st.sidebar.caption("No key needed — RAG & Chatbot work without one too.")


# HOME

if page == " Home":
    st.title(" Student AI/ML + NLP + RAG + GenAI Project")
    st.write(
        "This project uses **your own student dataset** (10 students) and "
        "applies the most important AI/ML ideas on top of it, step by step."
    )
    df = get_students()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", len(df))
    c2.metric("Average Math", f"{df['Math'].mean():.0f}")
    c3.metric("Average Attendance", f"{df['Attendance'].mean():.0f}%")
    c4.metric("Top Grade Count (A)", int((df["Grade"] == "A").sum()))
    st.dataframe(df, use_container_width=True)


# DATABASE

elif page == " Database":
    st.title(" Student Database (SQLite)")
    st.info(" A database stores rows permanently, unlike a Python list which "
            "disappears when the program stops. We use SQLite, Python's built-in database.")

    tab1, tab2, tab3 = st.tabs(["View", "Add Student", "Delete Student"])

    with tab1:
        st.dataframe(get_students(), use_container_width=True)
        st.caption("SQL used: `SELECT * FROM students`")

    with tab2:
        with st.form("add_form"):
            c1, c2 = st.columns(2)
            sid = c1.number_input("Student ID", min_value=111, value=111)
            name = c2.text_input("Name")
            age = c1.number_input("Age", 15, 25, 18)
            gender = c2.selectbox("Gender", ["Male", "Female"])
            math = c1.slider("Math", 0, 100, 75)
            science = c2.slider("Science", 0, 100, 75)
            english = c1.slider("English", 0, 100, 75)
            attendance = c2.slider("Attendance %", 0, 100, 90)
            grade = c1.selectbox("Grade", ["A", "B", "C"])
            feedback = st.text_area("Feedback", "Good course overall.")
            if st.form_submit_button("Add Student"):
                add_student({
                    "Student_ID": sid, "Name": name, "Age": age, "Gender": gender,
                    "Math": math, "Science": science, "English": english,
                    "Attendance": attendance, "Grade": grade, "Feedback": feedback,
                })
                st.success(f" {name} added!")
                st.caption("SQL used: `INSERT INTO students (...) VALUES (...)`")

    with tab3:
        df = get_students()
        choice = st.selectbox("Select student to delete",
                               df["Student_ID"].astype(str) + " - " + df["Name"])
        if st.button(" Delete"):
            delete_student(int(choice.split(" - ")[0]))
            st.success("Deleted. Refresh to see the updated list.")
            st.caption("SQL used: `DELETE FROM students WHERE Student_ID = ?`")


# EDA

elif page == " EDA":
    st.title(" Exploratory Data Analysis")
    st.info(" Before building any model, we look at the data first — averages, "
            "spread, and relationships between columns.")
    df = get_students()

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Grade Distribution")
        fig, ax = plt.subplots()
        df["Grade"].value_counts().plot(kind="bar", ax=ax, color="#4C72B0")
        st.pyplot(fig)
    with c2:
        st.subheader("Attendance vs Math Score")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df["Attendance"], df["Math"], color="#55A868")
        ax2.set_xlabel("Attendance %"); ax2.set_ylabel("Math Score")
        st.pyplot(fig2)

    st.subheader("Correlation Between Subjects")
    fig3, ax3 = plt.subplots()
    corr = df[["Math", "Science", "English", "Attendance"]].corr()
    cax = ax3.matshow(corr, cmap="coolwarm")
    fig3.colorbar(cax)
    ax3.set_xticks(range(len(corr.columns))); ax3.set_xticklabels(corr.columns, rotation=45)
    ax3.set_yticks(range(len(corr.columns))); ax3.set_yticklabels(corr.columns)
    st.pyplot(fig3)

    st.info(" **Data Preprocessing note:** Gender (Male/Female) is text, so before "
            "feeding it to a model we'd convert it to numbers (0/1) — this step is "
            "called **Encoding**. Our models below only use the number columns, "
            "so no encoding is needed here.")


# CLASSIFICATION

elif page == " Classification (predict Grade)":
    st.title(" Classification — Predict a Student's Grade")
    st.info(" **Supervised Learning:** the model learns from marks + attendance "
            "(input) and Grade (known answer) so it can predict the Grade of a "
            "NEW student. Predicting a category (A/B/C) is called Classification.")

    df = get_students()
    features = ["Math", "Science", "English", "Attendance"]
    X, y = df[features], df["Grade"]

    st.warning(" We only have 10 students, so accuracy will vary a lot each run. "
               "Add more students on the Database page to make the model smarter!")

    algo = st.selectbox("Choose algorithm", ["Decision Tree", "Logistic Regression"])
    if st.button(" Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        model = DecisionTreeClassifier(max_depth=3, random_state=1) if algo == "Decision Tree" \
            else LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.session_state["clf_model"] = model
        st.success(f" Accuracy on test students: **{acc*100:.0f}%**")

        st.subheader("Confusion Matrix (Actual vs Predicted)")
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, preds, labels=labels)
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha="center", va="center")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔮 Try a Live Prediction")
    c1, c2, c3, c4 = st.columns(4)
    m = c1.slider("Math", 0, 100, 80, key="cm")
    s = c2.slider("Science", 0, 100, 80, key="cs")
    e = c3.slider("English", 0, 100, 80, key="ce")
    a = c4.slider("Attendance", 0, 100, 90, key="ca")
    if st.button("Predict Grade"):
        if "clf_model" in st.session_state:
            new_data = pd.DataFrame([[m, s, e, a]], columns=features)
            st.subheader(f" Predicted Grade: **{st.session_state['clf_model'].predict(new_data)[0]}**")
        else:
            st.warning("Train a model first using the button above.")


# REGRESSION

elif page == " Regression (predict Science score)":
    st.title(" Regression — Predict a Number Instead of a Category")
    st.info(" **Supervised Learning too**, but instead of predicting a category "
            "(like Grade), Regression predicts a NUMBER — here, the Science score, "
            "using Math score and Attendance.")

    df = get_students()
    X, y = df[["Math", "Attendance"]], df["Science"]

    if st.button(" Train Regression Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        st.session_state["reg_model"] = model
        st.success(f" On average, predictions are off by about **{mae:.1f} marks** (Mean Absolute Error).")

        fig, ax = plt.subplots()
        ax.scatter(y_test, preds, color="#C44E52")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "--", color="gray")
        ax.set_xlabel("Actual Science Score"); ax.set_ylabel("Predicted Science Score")
        st.pyplot(fig)
        st.caption("Points closer to the dashed line = better predictions.")

    st.markdown("---")
    st.subheader(" Try a Live Prediction")
    c1, c2 = st.columns(2)
    m = c1.slider("Math score", 0, 100, 80, key="rm")
    a = c2.slider("Attendance %", 0, 100, 90, key="ra")
    if st.button("Predict Science Score"):
        if "reg_model" in st.session_state:
            pred = st.session_state["reg_model"].predict(pd.DataFrame([[m, a]], columns=["Math", "Attendance"]))[0]
            st.subheader(f" Predicted Science Score: **{pred:.0f}**")
        else:
            st.warning("Train the regression model first using the button above.")


# CLUSTERING

elif page == " Clustering (group students)":
    st.title(" Clustering — Grouping Students Without Labels")
    st.info(" **Unsupervised Learning:** unlike Classification/Regression, we do "
            "NOT tell the model any answers. It looks at Math, Science and English "
            "scores and groups similar students together by itself.")

    df = get_students()
    k = st.slider("Number of groups (clusters)", 2, 4, 2)

    if st.button(" Run Clustering"):
        X = df[["Math", "Science", "English"]]
        model = KMeans(n_clusters=k, n_init=10, random_state=1)
        df["Cluster"] = model.fit_predict(X)

        st.subheader("Result: Students Grouped by Performance")
        st.dataframe(df[["Name", "Math", "Science", "English", "Cluster"]], use_container_width=True)

        fig, ax = plt.subplots()
        ax.scatter(df["Math"], df["Science"], c=df["Cluster"], cmap="viridis", s=100)
        for _, row in df.iterrows():
            ax.annotate(row["Name"], (row["Math"], row["Science"]), fontsize=8)
        ax.set_xlabel("Math Score"); ax.set_ylabel("Science Score")
        st.pyplot(fig)
        st.caption("Students in the same color were grouped together by the algorithm — "
                   "no Grade labels were used at all!")


# NLP

elif page == " NLP - Feedback":
    st.title(" NLP — Reading Student Feedback")
    st.info(" NLP lets a computer understand text. Steps: **Clean → Tokenize → "
            "Remove filler words → Detect Sentiment**.")

    text = st.text_area("Try your own sentence", "The classes were great and I loved it!")
    if text:
        st.write("**Cleaned:**", clean_text(text))
        st.write("**Tokens:**", tokenize(text))
        st.write("**Sentiment:**", get_sentiment(text))

    st.markdown("---")
    st.subheader("Sentiment of All Student Feedback")
    df = get_students()
    df["Sentiment"] = df["Feedback"].apply(get_sentiment)
    st.dataframe(df[["Name", "Feedback", "Sentiment"]], use_container_width=True)

    fig, ax = plt.subplots()
    df["Sentiment"].value_counts().plot(kind="bar", ax=ax, color="#8172B2")
    st.pyplot(fig)


# RAG

elif page == " RAG - Q&A":
    st.title(" RAG — Retrieval Augmented Generation")
    st.info(" An AI doesn't know your school's private rules. RAG: "
            "**1) Retrieve** the right fact using TF-IDF + Cosine Similarity, "
            "**2) Augment** the question with that fact, **3) Generate** a natural answer.")

    knowledge_base = [
        "Students need at least 90% attendance to get an A grade.",
        "A grade means average marks above 90.",
        "B grade means average marks between 75 and 90.",
        "C grade means average marks between 60 and 75.",
        "The library is open from 8 AM to 6 PM on weekdays.",
    ]
    st.write("**Knowledge base:**", knowledge_base)

    query = st.text_input("Ask a question", "What attendance do I need for an A grade?")
    if st.button(" Search & Answer") and query:
        vec = TfidfVectorizer()
        vectors = vec.fit_transform(knowledge_base + [query])
        sims = cosine_similarity(vectors[-1], vectors[:-1])[0]
        best = knowledge_base[int(np.argmax(sims))]
        st.success(f"**Retrieved fact:** {best}")

        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model="claude-sonnet-5", max_tokens=150,
                    messages=[{"role": "user", "content":
                               f"Using only this fact: '{best}', answer: {query}"}]
                )
                answer = extract_text(resp)
                if answer:
                    st.write("**Generated answer:**", answer)
                else:
                    st.warning("The model returned no text content — showing the retrieved fact only.")
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.info("Showing the retrieved fact directly instead.")
        else:
            st.caption("Add a Claude API key in the sidebar for a naturally generated answer.")


# GENAI CHATBOT

elif page == " GenAI Chatbot":
    st.title(" GenAI Chatbot")
    st.info(" With an API key, Claude (a real LLM) generates fresh replies. "
            "Without one, a simple rule-based bot answers using keyword matching.")

    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": "Hi! Ask me about the project "}]

    def rule_bot(msg):
        msg = msg.lower()
        if "grade" in msg: return "Grades are A, B, or C based on average marks."
        if "attendance" in msg: return "You need 90%+ attendance for an A grade."
        if "hello" in msg or "hi" in msg: return "Hello! How can I help?"
        return "I'm a simple rule-based bot. Add an API key for smarter replies!"

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Type your message...")
    if user_msg:
        st.session_state.chat.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            if api_key:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)
                    resp = client.messages.create(
                        model="claude-sonnet-5", max_tokens=250,
                        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
                    )
                    reply = extract_text(resp)
                    if not reply:
                        reply = "(Model returned no text content) " + rule_bot(user_msg)
                except Exception as e:
                    st.error(f"API call failed: {e}")
                    reply = "(Falling back to rule-based bot) " + rule_bot(user_msg)
            else:
                reply = rule_bot(user_msg)
            st.write(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})
