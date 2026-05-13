import time
import streamlit as st
import json
from utils.medical_chat import DEFAULT_MODEL, extract_symptoms, prompt_checker, get_app_diagnosis, is_model_available

st.set_page_config(page_title="Symptom Checker", layout="centered")

st.title("Medical triage assistant")

def reset_assessment() -> None:
    for key in ("user_input", "extracted", "conditions", "urgency", "recommendation"):
        st.session_state.pop(key, None)

MODEL_OPTIONS = [
    DEFAULT_MODEL,
    "gemma4:e2b",
    "llama3.2:3b",
    "mistral:7b"
]

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Ollama model", MODEL_OPTIONS, index=0)
    model_name = selected_model
    model_installed = is_model_available(model_name)

    if not model_installed:
        st.warning(f'Model "{model_name}" is not downloaded. Run `ollama pull {model_name}` before using it.')


user_input = st.text_area(
    "Describe symptoms in plain language and the assistant will review the case and give a triage response.",
    placeholder="Example: I have had a sore throat, I am coughing a lot, and I feel very tired since yesterday.",
    height=160,
    key="user_input",
)

action_col, reset_col = st.columns([3, 1], gap="small")

with action_col:
    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
        disabled=not model_installed,
    )

with reset_col:
    restart_clicked = st.button("Restart", use_container_width=True)

if restart_clicked:
    reset_assessment()
    st.rerun()

if not model_installed:
    st.warning(f'Model "{model_name}" is not downloaded. Run `ollama pull {model_name}` before using it or open the sidebar to change the selected model.')

if analyze_clicked:
    if not user_input.strip():
        st.warning("Describe how you are feeling first.")
    else:
        with st.spinner("Extracting symptoms and generating assessment..."):

            extracted = extract_symptoms(model_name, user_input.strip())

            prompt_check = prompt_checker(model_name, extracted)
            if prompt_check.lower() != "yes":
                reset_assessment()
                st.warning("The extracted symptoms do not seem valid. Please adjust the input accordingly.")
                print(f"Prompt check response: {prompt_check}")
            else:
                diagnosis = get_app_diagnosis(model_name, extracted)

                # Parse the JSON string into a dictionary
                try:
                    diagnosis_dict = json.loads(diagnosis)
                except json.JSONDecodeError:
                    st.error("Model returned invalid JSON.")
                    st.code(diagnosis, language="json")
                    st.stop()


                st.session_state.extracted = extracted
                st.session_state.conditions = diagnosis_dict["conditions"]
                st.session_state.urgency = diagnosis_dict["urgency"]
                st.session_state.recommendation = diagnosis_dict["recommendation"]



if st.session_state.get("extracted"):
    st.subheader("Assessment")

    with st.container(border=True):
        st.caption("Extracted symptoms")
        st.write(st.session_state.extracted)

    left_col, right_col = st.columns([1.8, 1], gap="medium")

    with left_col:
        with st.container(border=True):
            st.markdown("#### Possible conditions")
            st.write(st.session_state.conditions)

    with right_col:
        with st.container(border=True):
            st.markdown("#### Urgency")
            urgency = st.session_state.urgency.strip().upper()
            if urgency == "LOW":
                st.success(urgency)
            elif urgency == "MEDIUM":
                st.warning(urgency)
            else:
                st.error(urgency)

    urgency_level = st.session_state.urgency.strip().upper()
    recommendation = st.session_state.recommendation.strip()
    if urgency_level == "LOW":
        st.info(recommendation)
    elif urgency_level == "MEDIUM":
        st.warning(recommendation)
    else:
        st.error(recommendation)


st.divider()
st.info(
    "This assistant is not a substitute for professional medical care. Seek urgent care for severe or worsening symptoms."
)
