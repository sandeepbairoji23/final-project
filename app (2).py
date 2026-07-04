"""
=====================================================================
 STUDENT AI/ML PROJECT  -  built on YOUR student data
=====================================================================
"""

import re
import io
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import anthropic   # KEEP (needed for Claude)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Student AI/ML Project", page_icon="🎓", layout="wide")

# ---------------- API KEY (FIXED SAFE VERSION) ----------------
st.sidebar.title("🎓 Menu")
page = st.sidebar.radio("Go to:", [
    "🏠 Home",
    "🗄️ Database",
    "📊 EDA",
    "🤖 Classification (predict Grade)",
    "📈 Regression (predict Science score)",
    "🧩 Clustering (group students)",
    "📝 NLP - Feedback",
    "🔍 RAG - Q&A",
    "💬 GenAI Chatbot",
    "📚 Topics Covered",
])

st.sidebar.markdown("---")

api_key = st.sidebar.text_input(
    "🔑 Claude API Key (optional)",
    type="password",
    help="Paste key starting with sk-ant-..."
)

st.sidebar.caption("No key needed — fallback bot works too.")


# ---------------- SAFE CLAUDE FUNCTION (NEW ADDITION) ----------------
def call_claude(api_key, messages, max_tokens=300):
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",   # FIXED MODEL
        max_tokens=max_tokens,
        messages=messages
    )

    # FIX: handle ThinkingBlock safely
    output = ""
    for block in response.content:
        if hasattr(block, "text"):
            output += block.text

    return output


# ---------------- YOUR ORIGINAL DATA ----------------
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

POSITIVE_WORDS = {"good","great","excellent","amazing","loved","helpful","fun","clear","wonderful","confident","satisfied","enjoyed"}
NEGATIVE_WORDS = {"struggled","confusing","difficult","stressful","heavy","bad","poor","hard"}

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
        return "Positive 😊"
    elif neg > pos:
        return "Negative 😞"
    return "Neutral 😐"


# ---------------- DATABASE ----------------
@st.cache_resource
def get_connection():
    return sqlite3.connect("students.db", check_same_thread=False)

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


# ---------------- HOME ----------------
if page == "🏠 Home":
    st.title("🎓 Student AI/ML Project")
    df = get_students()
    st.dataframe(df)


# ---------------- RAG (FIXED API PART ONLY) ----------------
elif page == "🔍 RAG - Q&A":

    knowledge_base = [
        "Students need at least 90% attendance to get an A grade.",
        "A grade means average marks above 90.",
        "B grade means average marks between 75 and 90.",
        "C grade means average marks between 60 and 75.",
        "The library is open from 8 AM to 6 PM on weekdays.",
    ]

    query = st.text_input("Ask a question")

    if st.button("Search & Answer") and query:

        vec = TfidfVectorizer()
        vectors = vec.fit_transform(knowledge_base + [query])
        sims = cosine_similarity(vectors[-1], vectors[:-1])[0]
        best = knowledge_base[int(np.argmax(sims))]

        st.success(best)

        if api_key and api_key.startswith("sk-ant-"):
            try:
                answer = call_claude(
                    api_key,
                    messages=[{
                        "role": "user",
                        "content": f"Use ONLY this fact: {best}. Answer: {query}"
                    }],
                    max_tokens=150
                )
                st.write(answer)

            except Exception as e:
                st.error(f"API error: {e}")
                st.write(best)
        else:
            st.warning("Invalid or missing API key")
            st.write(best)


# ---------------- CHATBOT (FIXED ONLY API PART) ----------------
elif page == "💬 GenAI Chatbot":

    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "assistant", "content": "Hi! Ask me anything 👋"}
        ]

    def rule_bot(msg):
        msg = msg.lower()
        if "grade" in msg:
            return "Grades are A, B, C."
        if "attendance" in msg:
            return "Need 90%+ for A grade."
        return "Rule bot active. Add API key for AI."

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Type message...")

    if user_msg:
        st.session_state.chat.append({"role": "user", "content": user_msg})

        with st.chat_message("assistant"):

            if api_key and api_key.startswith("sk-ant-"):
                try:
                    reply = call_claude(
                        api_key,
                        messages=st.session_state.chat,
                        max_tokens=250
                    )
                except Exception as e:
                    reply = f"API error: {e} " + rule_bot(user_msg)
            else:
                reply = rule_bot(user_msg)

            st.write(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})
