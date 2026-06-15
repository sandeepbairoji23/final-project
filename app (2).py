import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("Student Performance Dashboard")

try:
    students = pd.read_csv("Student_database.csv")
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

for col in ["Math", "Science", "English", "Attendance"]:
    students[col] = pd.to_numeric(students[col], errors="coerce")

students["Total"] = students["Math"] + students["Science"] + students["English"]
students["Average"] = students["Total"] / 3

option = st.sidebar.selectbox(
    "Select an option",
    [
        "Display Students",
        "Search Student",
        "Top Performer",
        "Subject Statistics",
        "Attendance Statistics",
        "Add Student",
        "EDA Graphs"
    ]
)

if option == "Display Students":
    st.subheader("All Students")
    st.dataframe(students)
    st.metric("Total Students", len(students))

elif option == "Search Student":
    st.subheader("Search Student")

    student_id = st.number_input("Enter Student ID", step=1)

    if st.button("Search"):
        found = students[students["Student_ID"] == student_id]

        if not found.empty:
            st.dataframe(found)
        else:
            st.warning("Student not found")

elif option == "Top Performer":
    st.subheader("Top Performer")

    best = students.loc[students["Total"].idxmax()]

    st.success(best["Name"])
    st.write(f"Total: {best['Total']}")
    st.write(f"Average: {best['Average']:.2f}")

elif option == "Subject Statistics":
    st.subheader("Subject Averages")

    col1, col2, col3 = st.columns(3)

    col1.metric("Math", round(students["Math"].mean(), 2))
    col2.metric("Science", round(students["Science"].mean(), 2))
    col3.metric("English", round(students["English"].mean(), 2))

elif option == "Attendance Statistics":
    st.subheader("Attendance Stats")

    st.metric("Average Attendance", round(students["Attendance"].mean(), 2))
    st.metric("Max Attendance", students["Attendance"].max())

    st.bar_chart(students.set_index("Name")["Attendance"])

elif option == "Add Student":
    st.subheader("Add Student")

    sid = st.number_input("Student ID", step=1)
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

        st.success("Student added successfully!")

elif option == "EDA Graphs":
    st.subheader("EDA Analysis")

    
    st.write("Subject Averages")

    subjects = ["Math", "Science", "English"]
    averages = [
        students["Math"].mean(),
        students["Science"].mean(),
        students["English"].mean()
    ]

    fig1, ax1 = plt.subplots()
    ax1.bar(subjects, averages)
    st.pyplot(fig1)

    st.markdown("---")

    
    st.write("Distribution")

    fig2, ax2 = plt.subplots()
    ax2.hist(students["Average"], bins=10)
    st.pyplot(fig2)

    st.markdown("---")

    
    st.write("Box Plot")

    fig3, ax3 = plt.subplots()

    data = [
        students["Math"].dropna(),
        students["Science"].dropna(),
        students["English"].dropna()
    ]

    ax3.boxplot(data, tick_labels=["Math", "Science", "English"])

    ax3.set_ylabel("Marks")
    ax3.set_title("Marks Distribution")

    st.pyplot(fig3)

    st.markdown("---")

    
    st.write("Attendance vs Average")

    fig4, ax4 = plt.subplots()
    ax4.scatter(students["Attendance"], students["Average"])
    st.pyplot(fig4)

    st.markdown("---")

    
    st.write("Correlation Heatmap")

    fig5, ax5 = plt.subplots()

    corr = students[["Math", "Science", "English", "Attendance", "Total", "Average"]].corr()

    sns.heatmap(corr, annot=True, ax=ax5, cmap="Blues")

    st.pyplot(fig5)

    st.markdown("---")

    
    st.write("Gender Distribution")

    fig6, ax6 = plt.subplots()
    gender = students["Gender"].value_counts()

    ax6.pie(gender.values, labels=gender.index, autopct="%1.1f%%")

    st.pyplot(fig6)

st.markdown("---")
st.write("Student Performance Management System")
