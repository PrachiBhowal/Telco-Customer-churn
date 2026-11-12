import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

model = load_model()

st.title("Customer Churn Predictor")
st.caption("Estimate how likely a telecom customer is to leave, and what to do about it.")

form_col, result_col = st.columns([1, 1], gap="large")

# ---------------- INPUTS ----------------
with form_col:
    with st.container(border=True):
        st.subheader("Account")
        a1, a2 = st.columns(2)
        tenure = a1.slider("Tenure (months)", 0, 72, 12)
        monthly = a2.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=1.0)
        contract = a1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = a2.selectbox("Payment method", ["Electronic check", "Mailed check",
                               "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = a1.selectbox("Paperless billing", ["Yes", "No"])

    with st.container(border=True):
        st.subheader("Services")
        s1, s2 = st.columns(2)
        internet = s1.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        phone = s2.selectbox("Phone service", ["Yes", "No"])
        multi = s1.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        sec = s2.selectbox("Online security", ["No", "Yes", "No internet service"])
        backup = s1.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device = s2.selectbox("Device protection", ["No", "Yes", "No internet service"])
        tech = s1.selectbox("Tech support", ["No", "Yes", "No internet service"])
        tv = s2.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = s1.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

    with st.container(border=True):
        st.subheader("Customer")
        d1, d2 = st.columns(2)
        gender = d1.selectbox("Gender", ["Female", "Male"])
        senior = d2.selectbox("Senior citizen", ["No", "Yes"])
        partner = d1.selectbox("Partner", ["Yes", "No"])
        dependents = d2.selectbox("Dependents", ["Yes", "No"])

# ---------------- PREDICTION (updates live) ----------------
row = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner, "Dependents": dependents, "tenure": tenure,
    "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
    "OnlineSecurity": sec, "OnlineBackup": backup, "DeviceProtection": device,
    "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": tenure * monthly}])

p = float(model.predict_proba(row)[0, 1])
level, color = (("High", "#E74C3C") if p >= 0.6 else
                ("Medium", "#F39C12") if p >= 0.35 else ("Low", "#2ECC71"))

with result_col:
    with st.container(border=True):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100,
            number={"suffix": "%", "font": {"size": 52}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                   "steps": [{"range": [0, 35], "color": "rgba(46,204,113,0.18)"},
                             {"range": [35, 60], "color": "rgba(243,156,18,0.18)"},
                             {"range": [60, 100], "color": "rgba(231,76,60,0.18)"}]}))
        fig.update_layout(height=260, margin=dict(t=20, b=0, l=30, r=30),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<h3 style='text-align:center;color:{color};margin-top:-10px'>"
                    f"{level} churn risk</h3>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Risk factors")
        flags = []
        if contract == "Month-to-month": flags.append("Month-to-month contract")
        if tenure < 12: flags.append(f"Short tenure ({tenure} months)")
        if internet == "Fiber optic": flags.append("Fiber-optic plan")
        if payment == "Electronic check": flags.append("Pays by electronic check")
        if internet != "No" and tech == "No": flags.append("No tech support")
        if internet != "No" and sec == "No": flags.append("No online security")
        if monthly > 80: flags.append(f"High monthly charges (${monthly:.0f})")
        if flags:
            for f in flags:
                st.markdown(f"- {f}")
        else:
            st.markdown("No major risk factors found.")

    with st.container(border=True):
        st.subheader("Recommended action")
        if level == "High":
            st.error("Contact proactively. Offer a discounted annual contract or a free tech-support/security bundle.")
        elif level == "Medium":
            st.warning("Keep an eye on this account. Send a loyalty offer or encourage automatic payment.")
        else:
            st.success("No action needed.")