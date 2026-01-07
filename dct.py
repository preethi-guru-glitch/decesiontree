import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# App title
st.title("🌳 Decision Tree Classifier Demo")

# Sample dataset (Study Hours vs Pass/Fail)
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [60, 55, 65, 70, 75, 80, 85, 90, 95, 98],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

st.subheader("📊 Dataset")
st.write(df)

# Features and target
X = df[["Hours_Studied", "Attendance"]]
y = df["Pass"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree Model
model = DecisionTreeClassifier(criterion="gini", max_depth=3)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
st.subheader("📈 Model Evaluation")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# User input
st.subheader("🔮 Predict Pass or Fail")
hours = st.number_input("Enter study hours:", min_value=0, max_value=24)
attendance = st.number_input("Enter attendance percentage:", min_value=0, max_value=100)

if st.button("Predict"):
    prediction = model.predict([[hours, attendance]])
    if prediction[0] == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("❌ Student will FAIL")
