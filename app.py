import streamlit as st
import pandas as pd
import joblib

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Employee Burnout & Attrition Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("burnout_model.joblib")

model = load_model()

# ===============================
# Feature Order (CRITICAL)
# ===============================
FEATURE_ORDER = [
    "tenure_months",
    "stress_level",
    "career_progression_score",
    "collaboration_score",
    "workload_score",
    "satisfaction_score",
    "team_sentiment",
    "overtime_hours",
    "salary"
]

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.title("👤 Employee Attributes")

daily_working_hours = st.sidebar.slider(
    "Daily Working Hours", 4, 14, 8
)

salary = st.sidebar.number_input(
    "Monthly Salary (₹)", min_value=10000, max_value=500000, value=30000, step=5000
)

job_level = st.sidebar.selectbox(
    "Job Level", ["Entry", "Junior", "Mid", "Senior", "Manager"]
)

department = st.sidebar.selectbox(
    "Department", ["IT", "HR", "Finance", "Sales", "Research & Development"]
)

st.sidebar.markdown("---")

workload_score = st.sidebar.slider(
    "Workload Score (1–10)", 1, 10, 6
)

satisfaction_score = st.sidebar.slider(
    "Job Satisfaction (1–10)", 1, 10, 5
)

team_sentiment = st.sidebar.slider(
    "Team Sentiment (1–10)", 1, 10, 6
)

# ===============================
# Main Content
# ===============================
st.title("🧠 Employee Burnout & Attrition Risk Dashboard")

st.markdown("""
This dashboard provides **decision-support analytics** for HR and leadership teams.  
Predictions are **probabilistic**, not final judgments.
""")

st.info("👉 Enter employee details from the left panel and click **Analyze Burnout Risk**")

# ===============================
# Rating Guide
# ===============================
with st.expander("📘 How to Rate Employees (Read Before Scoring)", expanded=True):
    st.markdown("""
**Workload (1–10)**  
• 1–3 → Light workload  
• 4–6 → Balanced workload  
• 7–10 → High pressure / overtime  

**Job Satisfaction (1–10)**  
• 1–3 → Dissatisfied  
• 4–6 → Neutral  
• 7–10 → Highly satisfied  

**Team Sentiment (1–10)**  
• 1–3 → Conflict / low morale  
• 4–6 → Mixed collaboration  
• 7–10 → Strong teamwork  
""")

# ===============================
# Analyze Button
# ===============================
analyze = st.button("🔍 Analyze Burnout Risk", use_container_width=True)

if analyze:
    overtime_hours = max(daily_working_hours - 8, 0)

    input_df = pd.DataFrame([{
        "tenure_months": 24,  # default neutral tenure
        "stress_level": workload_score,
        "career_progression_score": satisfaction_score,
        "collaboration_score": team_sentiment,
        "workload_score": workload_score,
        "satisfaction_score": satisfaction_score,
        "team_sentiment": team_sentiment,
        "overtime_hours": overtime_hours,
        "salary": salary
    }])

    # 🔥 FIX: FORCE CORRECT FEATURE ORDER
    input_df = input_df[FEATURE_ORDER]

    prediction = model.predict(input_df.values)[0]

    probability = model.predict_proba(input_df.values)[0][prediction]


    # ===============================
    # Results
    # ===============================
    st.markdown("## 📊 Burnout Risk Analysis")

    if prediction == 2:
        st.error(f"**High Burnout Risk** 🔴  \nProbability: {probability:.2%}")
        st.write("Recommended Action: Immediate intervention, workload review, and manager discussion.")
    elif prediction == 1:
        st.warning(f"**Medium Burnout Risk** 🟠  \nProbability: {probability:.2%}")
        st.write("Recommended Action: Monitor trends and engage proactively.")
    else:
        st.success(f"**Low Burnout Risk** 🟢  \nProbability: {probability:.2%}")
        st.write("Recommended Action: Maintain current workload and engagement.")

    st.markdown("---")

    # ===============================
    # Feedback Buttons
    # ===============================
    st.markdown("### 🧪 HR Feedback (Optional)")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🟢 Prediction Correct"):
            st.success("Feedback saved.")

    with col2:
        if st.button("🔴 Prediction Incorrect"):
            st.warning("Feedback saved for retraining.")

    with col3:
        if st.button("🔵 Skip / No Opinion"):
            st.info("Feedback skipped.")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Built for HR analytics, workforce planning, and leadership decision support.")
