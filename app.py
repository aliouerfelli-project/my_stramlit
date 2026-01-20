import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="IDS", layout="centered")

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🛡️ Intrusion Detection System")

file = st.file_uploader("Upload Test_data.csv", type="csv")

if file:
    df = pd.read_csv(file)

    X = df.drop("class", axis=1) if "class" in df.columns else df
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    df["Label"] = ["🚨 Anomaly" if p == 1 else "✅ Normal" for p in preds]

    st.write("### Résultat")
    st.metric("Total", len(df))
    st.metric("Anomalies", int(sum(preds)))

    fig = px.pie(df, names="Label", hole=0.4)
    st.plotly_chart(fig)

    st.dataframe(df[["Label"]].head(20))
