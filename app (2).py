import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Dashboard + ML", layout="wide")
st.title("Student Performance Dashboard + ML")

try:
    students = pd.read_csv("Student_database.csv")
except:
    st.error("Error loading file")
    st.stop()

students["Math"] = pd.to_numeric(students["Math"], errors="coerce")
students["Science"] = pd.to_numeric(students["Science"], errors="coerce")
students["English"] = pd.to_numeric(students["English"], errors="coerce")
students["Attendance"] = pd.to_numeric(students["Attendance"], errors="coerce")

students = students.dropna()

students["Total"] = students["Math"] + students["Science"] + students["English"]
students["Average"] = students["Total"] / 3
students["Result"] = students["Average"].apply(lambda x: 1 if x >= 40 else 0)

def train_model(data):
    X = data[["Math", "Science", "English", "Attendance"]]
    y = data["Result"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    return model, acc

model, acc = train_model(students)

st.sidebar.success(f"Accuracy {acc:.2f}")

option = st.sidebar.selectbox(
    "Select",
    ["Display Students", "Search Student", "Top Performer", "Subject Statistics", "Attendance Statistics", "Add Student", "EDA Graphs", "ML Prediction"]
)

if option == "Display Students":
    st.dataframe(students)
    st.metric("Total Students", len(students))

elif option == "Search Student":
    sid = st.number_input("ID", step=1)
    if st.button("Search"):
        res = students[students["Student_ID"] == sid]
        if not res.empty:
            st.dataframe(res)
        else:
            st.warning("Not found")

elif option == "Top Performer":
    best = students.loc[students["Total"].idxmax()]
    st.success(best["Name"])
    st.write(best["Total"])
    st.write(f"{best['Average']:.2f}")

elif option == "Subject Statistics":
    col1, col2, col3 = st.columns(3)
    col1.metric("Math", round(students["Math"].mean(), 2))
    col2.metric("Science", round(students["Science"].mean(), 2))
    col3.metric("English", round(students["English"].mean(), 2))

elif option == "Attendance Statistics":
    st.metric("Avg Attendance", round(students["Attendance"].mean(), 2))
    st.metric("Max Attendance", students["Attendance"].max())
    st.bar_chart(students.set_index("Name")["Attendance"])

elif option == "Add Student":
    sid = st.number_input("ID", step=1)
    name = st.text_input("Name")
    age = st.number_input("Age", step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    math = st.number_input("Math", 0, 100)
    science = st.number_input("Science", 0, 100)
    english = st.number_input("English", 0, 100)
    attendance = st.number_input("Attendance", 0, 100)

    if st.button("Add"):
        new = pd.DataFrame([{
            "Student_ID": sid,
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Math": math,
            "Science": science,
            "English": english,
            "Attendance": attendance
        }])

        students = pd.concat([students, new], ignore_index=True)
        students.to_csv("Student_database.csv", index=False)
        st.success("Added")

elif option == "EDA Graphs":
    fig, ax = plt.subplots()
    ax.bar(["Math", "Science", "English"], [students["Math"].mean(), students["Science"].mean(), students["English"].mean()])
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.hist(students["Average"], bins=10)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.boxplot([students["Math"], students["Science"], students["English"]])
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.scatter(students["Attendance"], students["Average"])
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(students[["Math", "Science", "English", "Attendance", "Total", "Average"]].corr(), ax=ax, annot=True)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    gender = students["Gender"].value_counts()
    ax.pie(gender.values, labels=gender.index, autopct="%1.1f%%")
    st.pyplot(fig)

elif option == "ML Prediction":
    math = st.number_input("Math", 0, 100)
    science = st.number_input("Science", 0, 100)
    english = st.number_input("English", 0, 100)
    attendance = st.number_input("Attendance", 0, 100)

    if st.button("Predict"):
        inp = [[math, science, english, attendance]]
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1]

        if pred == 1:
            st.success("PASS")
        else:
            st.error("FAIL")

        st.write(prob)
