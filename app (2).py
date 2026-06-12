import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Title
st.title("Student Performance Dashboard")

# Load the CSV file
try:
    students = pd.read_csv("Student_database.csv")
except:
    st.error("CSV file not found!")
    st.stop()

# Calculate total marks
students["Total"] = students["Math"] + students["Science"] + students["English"]

# Calculate average
students["Average"] = students["Total"] / 3

# Create sidebar menu
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

# Display Students
if option == "Display Students":
    st.subheader("All Students")
    st.dataframe(students)
    st.metric("Total Number of Students", len(students))

# Search Student
elif option == "Search Student":
    st.subheader("Search for a Student")
    
    student_id = st.number_input("Enter Student ID", step=1)
    
    if st.button("Search"):
        found = students[students["Student_ID"] == student_id]
        
        if len(found) > 0:
            st.dataframe(found)
        else:
            st.warning("Student not found")

# Top Performer
elif option == "Top Performer":
    st.subheader("Top Performer")
    
    # Find student with highest total marks
    best_student = students.loc[students["Total"].idxmax()]
    
    st.success(f"Student Name: {best_student['Name']}")
    st.write(f"Total Marks: {best_student['Total']}")
    st.write(f"Average: {best_student['Average']:.2f}")

# Subject Statistics
elif option == "Subject Statistics":
    st.subheader("Subject Wise Averages")
    
    # Calculate average for each subject
    math_average = students["Math"].mean()
    science_average = students["Science"].mean()
    english_average = students["English"].mean()
    
    # Display in three columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Math Average", round(math_average, 2))
    col2.metric("Science Average", round(science_average, 2))
    col3.metric("English Average", round(english_average, 2))
    
    # Display as bar chart
    chart_data = pd.DataFrame(
        {
            "Average": [math_average, science_average, english_average]
        },
        index=["Math", "Science", "English"]
    )
    st.bar_chart(chart_data)

# Attendance Statistics
elif option == "Attendance Statistics":
    st.subheader("Attendance Statistics")
    
    avg_attendance = students["Attendance"].mean()
    max_attendance = students["Attendance"].max()
    
    st.metric("Average Attendance", round(avg_attendance, 2))
    st.metric("Highest Attendance", max_attendance)
    
    # Bar chart for each student attendance
    attendance_chart = students.set_index("Name")["Attendance"]
    st.bar_chart(attendance_chart)

# Add Student
elif option == "Add Student":
    st.subheader("Add a New Student")
    
    # Input fields
    student_id = st.number_input("Student ID", step=1)
    name = st.text_input("Student Name")
    age = st.number_input("Age", step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    math_marks = st.number_input("Math Marks", 0, 100)
    science_marks = st.number_input("Science Marks", 0, 100)
    english_marks = st.number_input("English Marks", 0, 100)
    attendance = st.number_input("Attendance", 0, 100)
    
    # Add button
    if st.button("Add Student"):
        # Create new row
        new_row = pd.DataFrame([{
            "Student_ID": student_id,
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Math": math_marks,
            "Science": science_marks,
            "English": english_marks,
            "Attendance": attendance
        }])
        
        # Add to dataframe
        students = pd.concat([students, new_row], ignore_index=True)
        
        # Save to CSV
        students.to_csv("Student_database.csv", index=False)
        
        st.success("Student added successfully!")

# EDA Graphs
elif option == "EDA Graphs":
    st.subheader("Data Analysis Graphs")
    
    # Graph 1: Subject Average Comparison
    st.write("## 1. Subject Average Comparison")
    
    fig1, ax1 = plt.subplots()
    subjects = ["Math", "Science", "English"]
    averages = [students["Math"].mean(), students["Science"].mean(), students["English"].mean()]
    
    ax1.bar(subjects, averages, color=['blue', 'green', 'orange'])
    ax1.set_ylabel("Average Marks")
    ax1.set_title("Subject Averages")
    st.pyplot(fig1)
    
    st.markdown("---")
    
    # Graph 2: Marks Distribution
    st.write("## 2. Marks Distribution (Histogram)")
    
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].hist(students["Math"], bins=8, color='blue', edgecolor='black')
    axes[0].set_title("Math Marks Distribution")
    axes[0].set_xlabel("Marks")
    axes[0].set_ylabel("Number of Students")
    
    axes[1].hist(students["Science"], bins=8, color='green', edgecolor='black')
    axes[1].set_title("Science Marks Distribution")
    axes[1].set_xlabel("Marks")
    axes[1].set_ylabel("Number of Students")
    
    axes[2].hist(students["English"], bins=8, color='orange', edgecolor='black')
    axes[2].set_title("English Marks Distribution")
    axes[2].set_xlabel("Marks")
    axes[2].set_ylabel("Number of Students")
    
    st.pyplot(fig2)
    
    st.markdown("---")
    
    # Graph 3: Box Plot
    st.write("## 3. Marks Spread Comparison (Box Plot)")
    
    fig3, ax3 = plt.subplots()
    data_for_box = [students["Math"], students["Science"], students["English"]]
    ax3.boxplot(data_for_box, labels=["Math", "Science", "English"])
    ax3.set_ylabel("Marks")
    ax3.set_title("Marks Distribution Box Plot")
    st.pyplot(fig3)
    
    st.markdown("---")
    
    # Graph 4: Scatter Plot (Attendance vs Average)
    st.write("## 4. Attendance vs Average Marks")
    
    fig4, ax4 = plt.subplots()
    ax4.scatter(students["Attendance"], students["Average"], color='blue', s=100)
    ax4.set_xlabel("Attendance %")
    ax4.set_ylabel("Average Marks")
    ax4.set_title("Attendance vs Average Marks")
    st.pyplot(fig4)
    
    st.markdown("---")
    
    # Graph 5: Correlation Heatmap
    st.write("## 5. Correlation Heatmap")
    
    fig5, ax5 = plt.subplots()
    correlation = students[["Math", "Science", "English", "Attendance", "Total", "Average"]].corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", ax=ax5, cmap="Blues")
    st.pyplot(fig5)
    
    st.markdown("---")
    
    # Graph 6: Gender Distribution
    st.write("## 6. Gender Distribution (Pie Chart)")
    
    fig6, ax6 = plt.subplots()
    gender_counts = students["Gender"].value_counts()
    ax6.pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%")
    ax6.set_title("Gender Distribution")
    st.pyplot(fig6)

# Footer
st.markdown("---")
st.write("Student Performance Management System")
