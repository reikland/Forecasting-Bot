import asyncio
import streamlit as st

from forecasting import run_pipeline


st.set_page_config(page_title="Forecasting Bot", layout="wide")
st.title("Forecasting Bot")

with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")
    model_name = st.text_input("OpenRouter Model Name", value="")

st.header("Question")
question_title = st.text_input("Question title")
context = st.text_area("Context / info dump / resolution criteria", height=200)

run_button = st.button("Run Forecast")

if run_button:
    if not api_key or not model_name or not question_title:
        st.error("Please provide API key, model name, and question title.")
    else:
        with st.spinner("Running forecasting pipeline..."):
            try:
                result = asyncio.run(run_pipeline(api_key, model_name, question_title, context))
                st.subheader("Supreme Judge Decision")
                st.write(result.supreme_decision)
            except Exception as exc:
                st.error(f"Error: {exc}")
