import streamlit as st
import pandas as pd
import numpy as np

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

# Optional GenAI (OpenAI)
import openai

# -------------------------
# UI CONFIG
# -------------------------
st.set_page_config(page_title="FULL AI SYSTEM", layout="wide")
st.title("🚀 ALL-IN-ONE AI/ML + NLP + RAG + GENAI SYSTEM")

# -------------------------
# SESSION STATE
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------
# FILE UPLOAD
# -------------------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📊 Dataset")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns

    # -------------------------
    # EDA
    # -------------------------
    st.subheader("📈 EDA")
    if len(numeric_cols) > 0:
        col = st.selectbox("Select column", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    # -------------------------
    # ML MODEL
    # -------------------------
    st.subheader("🤖 MACHINE LEARNING (Random Forest)")

    target = st.selectbox("Target column", df.columns)

    if st.button("Train ML Model"):
        X = pd.get_dummies(df.drop(columns=[target]))
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        st.success("Accuracy: " + str(accuracy_score(y_test, pred)))

    # -------------------------
    # NEURAL NETWORK (MLP)
    # -------------------------
    st.subheader("🧠 NEURAL NETWORK (MLPClassifier)")

    if st.button("Train Neural Network"):
        X = pd.get_dummies(df.drop(columns=[target]))
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
        nn.fit(X_train, y_train)

        pred = nn.predict(X_test)

        st.success("NN Accuracy: " + str(accuracy_score(y_test, pred)))

    # -------------------------
    # NLP MODULE
    # -------------------------
    st.subheader("📝 NLP (Text Similarity Engine)")

    text_col = st.selectbox("Select TEXT column", df.columns)

    if st.button("Run NLP Engine"):
        texts = df[text_col].astype(str).tolist()

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(texts)

        query = st.text_input("Ask query from dataset text")

        if query:
            q_vec = vectorizer.transform([query])
            sim = cosine_similarity(q_vec, tfidf)

            idx = np.argmax(sim)
            st.write("Most similar text:")
            st.success(texts[idx])

    # -------------------------
    # RAG (Simple Retrieval System)
    # -------------------------
    st.subheader("📚 RAG (Retrieval Augmented Generation - Simple)")

    rag_col = st.selectbox("Select RAG TEXT column", df.columns)

    if st.button("Build RAG Index"):
        st.session_state.docs = df[rag_col].astype(str).tolist()

        vectorizer = TfidfVectorizer()
        st.session_state.vec = vectorizer.fit_transform(st.session_state.docs)
        st.session_state.vectorizer = vectorizer

        st.success("RAG index built!")

    query = st.text_input("Ask RAG question")

    if query and "docs" in st.session_state:
        q_vec = st.session_state.vectorizer.transform([query])
        sim = cosine_similarity(q_vec, st.session_state.vec)

        idx = np.argmax(sim)
        st.write("📌 Retrieved Answer:")
        st.info(st.session_state.docs[idx])

    # -------------------------
    # GENAI CHATBOT
    # -------------------------
    st.subheader("🤖 GENAI CHATBOT")

    api_key = st.text_input("OpenAI API Key", type="password")

    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])

    user_msg = st.chat_input("Ask AI anything...")

    if user_msg:
        st.session_state.chat.append({"role": "user", "content": user_msg})
        st.chat_message("user").write(user_msg)

        if api_key:
            try:
                openai.api_key = api_key

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.chat
                )

                bot = response["choices"][0]["message"]["content"]

            except:
                bot = "Error with OpenAI API"

        else:
            bot = "Enter API key to use GenAI chatbot"

        st.session_state.chat.append({"role": "assistant", "content": bot})
        st.chat_message("assistant").write(bot)

else:
    st.warning("Upload dataset to activate AI system")
