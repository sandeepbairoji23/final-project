import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Student Dashboard + ML", layout="wide")
st.title("📊 Student Performance Dashboard + ML")

# =========================
# LOAD DATA
# =========================
try:
    students = pd.read_csv("Student_database.csv")
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# =========================
# DATA CLEANING + FEATURES
# =========================
students["Math"] = pd.to_numeric(students["Math"], errors="coerce")
students["Science"] = pd.to_numeric(students["Science"], errors="coerce")
students["English"] = pd.to_numeric(students["English"], errors="coerce")
students["Attendance"] = pd.to_numeric(students["Attendance"], errors="coerce")

students = students.dropna()

students["Total"] = students["Math"] + students["Science"] + students["English"]
students["Average"] = students["Total"] / 3

# Target column for ML (1 = Pass, 0 = Fail)
students["Result"] = students["Average"].apply(lambda x: 1 if x >= 40 else 0)

# =========================
# ML MODEL TRAINING
# =========================
def train_model(data):
    X = data[["Math", "Science", "English", "Attendance"]]
    y = data["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc

model, accuracy = train_model(students)

st.sidebar.success(f"🤖 Model Accuracy: {accuracy:.2f}")

# =========================
# SIDEBAR MENU
# =========================
option = st.sidebar.selectbox(
    "Select Feature",
    [
        "Display Students",
        "Search Student",
        "Top Performer",
        "Subject Statistics",
        "Attendance Statistics",
        "Add Student",
        "EDA Graphs",
        "ML Prediction"
    ]
)

# =========================
# DISPLAY STUDENTS
# =========================
if option == "Display Students":
    st.subheader("All Students Data")
    st.dataframe(students)
    st.metric("Total Students", len(students))

# =========================
# SEARCH STUDENT
# =========================
elif option == "Search Student":
    st.subheader("Search Student")

    student_id = st.number_input("Enter Student ID", step=1)

    if st.button("Search"):
        result = students[students["Student_ID"] == student_id]

        if not result.empty:
            st.dataframe(result)
        else:
            st.warning("Student not found")

# =========================
# TOP PERFORMER
# =========================
elif option == "Top Performer":
    st.subheader("Top Performer")

    best = students.loc[students["Total"].idxmax()]

    st.success(best["Name"])
    st.write(f"Total: {best['Total']}")
    st.write(f"Average: {best['Average']:.2f"])

# =========================
# SUBJECT STATS
# =========================
elif option == "Subject Statistics":
    st.subheader("Subject Averages")

    col1, col2, col3 = st.columns(3)

    col1.metric("Math", round(students["Math"].mean(), 2))
    col2.metric("Science", round(students["Science"].mean(), 2))
    col3.metric("English", round(students["English"].mean(), 2))

# =========================
# ATTENDANCE
# =========================
elif option == "Attendance Statistics":
    st.subheader("Attendance Stats")

    st.metric("Average Attendance", round(students["Attendance"].mean(), 2))
    st.metric("Max Attendance", students["Attendance"].max())

    st.bar_chart(students.set_index("Name")["Attendance"])

# =========================
# ADD STUDENT
# =========================
elif option == "Add Student":
    st.subheader("Add New Student")

    sid = st.number_input("Student ID", step=1)
    name = st.text_input("Name")
    age = st.number_input("Age", step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])

    math = st.number_input("Math", 0, 100)
    science = st.number_input("Science", 0, 100)
    english = st.number_input("English", 0, 100)
    attendance = st.number_input("Attendance", 0, 100)

    if st.button("Add Student"):
        if name == "":
            st.error("Name cannot be empty")
        elif sid in students["Student_ID"].values:
            st.error("Student ID already exists")
        else:
            new_student = pd.DataFrame([{
                "Student_ID": sid,
                "Name": name,
                "Age": age,
                "Gender": gender,
                "Math": math,
                "Science": science,
                "English": english,
                "Attendance": attendance
            }])

            students = pd.concat([students, new_student], ignore_index=True)
            students.to_csv("Student_database.csv", index=False)

            st.success("Student added successfully!")

# =========================
# EDA GRAPHS
# =========================
elif option == "EDA Graphs":
    st.subheader("Exploratory Data Analysis")

    # Subject averages
    fig1, ax1 = plt.subplots()
    ax1.bar(["Math", "Science", "English"],
            [students["Math"].mean(),
             students["Science"].mean(),
             students["English"].mean()])
    st.pyplot(fig1)

    st.markdown("---")

    # Distribution
    fig2, ax2 = plt.subplots()
    ax2.hist(students["Average"], bins=10)
    st.pyplot(fig2)

    st.markdown("---")

    # Boxplot
    fig3, ax3 = plt.subplots()
    ax3.boxplot([
        students["Math"],
        students["Science"],
        students["English"]
    ], tick_labels=["Math", "Science", "English"])
    st.pyplot(fig3)

    st.markdown("---")

    # Scatter
    fig4, ax4 = plt.subplots()
    ax4.scatter(students["Attendance"], students["Average"])
    st.pyplot(fig4)

    st.markdown("---")

    # Correlation heatmap
    fig5, ax5 = plt.subplots()
    sns.heatmap(students[["Math", "Science", "English", "Attendance", "Total", "Average"]].corr(),
                annot=True, ax=ax5)
    st.pyplot(fig5)

    st.markdown("---")

    # Gender pie chart
    fig6, ax6 = plt.subplots()
    gender = students["Gender"].value_counts()
    ax6.pie(gender.values, labels=gender.index, autopct="%1.1f%%")
    st.pyplot(fig6)

# =========================
# ML PREDICTION
# =========================
elif option == "ML Prediction":
    st.subheader("🤖 Student Performance Prediction")

    math = st.number_input("Math", 0, 100)
    science = st.number_input("Science", 0, 100)
    english = st.number_input("English", 0, 100)
    attendance = st.number_input("Attendance", 0, 100)

    if st.button("Predict"):
        input_data = [[math, science, english, attendance]]

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success("🎉 Student will PASS")
        else:
            st.error("⚠️ Student may FAIL")

        st.write(f"Pass Probability: {probability:.2f}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.write("📌 Student Performance Management System + ML Model")```

---

# 🔥 What you now have

You now built a:

### ✔ Data Dashboard
### ✔ Data Cleaning Pipeline
### ✔ Feature Engineering
### ✔ ML Model (Random Forest)
### ✔ Real-time Prediction System
### ✔ EDA Visualizations

---

# 🚀 If you want next upgrade (important for job)

I can help you add:

### 🔥 1. Save ML model (joblib)
### 🔥 2. Login system (admin panel)
### 🔥 3. SQLite database (replace CSV)
### 🔥 4. Deploy on Streamlit Cloud
### 🔥 5. Resume project description (very important)

Just tell 👍
