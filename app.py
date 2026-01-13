import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Decision Intelligence",
    page_icon="💳",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .hero {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 16px;
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .approved {
        color: #15803d;
        font-weight: 800;
        font-size: 1.8rem;
    }
    .rejected {
        color: #b91c1c;
        font-weight: 800;
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown('<div class="hero">Loan Decision Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">AI-powered credit approval system with explainability</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Applicant Information")

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Married = st.sidebar.selectbox("Married", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.sidebar.divider()

ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.sidebar.number_input("Loan Term (months)", min_value=0)
Credit_History = st.sidebar.selectbox("Credit History", [1.0, 0.0])

predict = st.sidebar.button("🚀 Run Credit Decision", use_container_width=True)

# ---------------- MAIN AREA ----------------
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Decision Summary")
    st.write(
        "This system evaluates loan applications using a trained machine learning "
        "model based on historical approval patterns."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Info")
    st.metric("Model Type", "Random Forest")
    st.metric("Training Data", "600+ Applications")
    st.metric("Explainability", "SHAP-ready")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict:
    payload = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area
    }

    with st.spinner("Running credit risk analysis..."):
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    if response.status_code == 200:
        result = response.json()

        st.divider()
        st.subheader("Loan Decision")

        decision_col, prob_col = st.columns(2)

        with decision_col:
            if result["prediction"] == "Approved":
                st.markdown("<div class='approved'>✅ APPROVED</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='rejected'>❌ REJECTED</div>", unsafe_allow_html=True)

        with prob_col:
            st.metric(
                "Approval Probability",
                f"{int(result['approval_probability'] * 100)}%"
            )
            st.progress(result["approval_probability"])

        # Explanation placeholder (important)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # ---------------- EXPLAINABILITY ----------------
st.divider()
st.subheader("📊 Decision Explanation")

features = result["feature_importance"]
input_vals = result["input_values"]

df = pd.DataFrame(features, columns=["Feature", "Importance"])

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 🔍 Top Influencing Features")

    fig, ax = plt.subplots()
    ax.barh(df["Feature"], df["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Impact on Decision")

    st.pyplot(fig)

with col2:
    st.markdown("### 📈 Applicant vs Typical Approved User")

    comparison_data = {
        "Your Value": [
            input_vals["Credit_History"],
            input_vals["ApplicantIncome"],
            input_vals["LoanAmount"]
        ],
        "Typical Approved": [1.0, 5000, 150]
    }

    comp_df = pd.DataFrame(
        comparison_data,
        index=["Credit History", "Income", "Loan Amount"]
    )

    st.dataframe(comp_df, use_container_width=True)

# ---------------- TEXTUAL EXPLANATION ----------------
st.markdown("### 🧠 What influenced this decision?")

reasons = []
if input_vals["Credit_History"] == 1.0:
    reasons.append("✅ Strong credit history significantly improved approval chances.")
else:
    reasons.append("❌ Missing credit history negatively impacted the decision.")

if input_vals["ApplicantIncome"] > 4000:
    reasons.append("✅ Applicant income is above average, increasing trustworthiness.")
else:
    reasons.append("⚠️ Lower income reduced approval confidence.")

if input_vals["LoanAmount"] > 200:
    reasons.append("⚠️ Requested loan amount is relatively high.")

for r in reasons:
    st.write(r)
