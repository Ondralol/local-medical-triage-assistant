import ollama

DEFAULT_MODEL = "qwen2.5:0.5b"

EXTRACT_PROMPT = """
You are a symptom extraction tool.
Extract ONLY the symptoms, injuries, or physical complaints explicitly stated by the user.
Do NOT infer, assume, or add anything not directly mentioned.
Return a comma-separated list only. No explanation, no preamble.
If no symptoms are mentioned, return "none".
"""

DIAGNOSE_PROMPT = """
You are a medical triage assistant. Given a list of symptoms and the top RAG and BM25 results and urgency level, respond with:

IMPORTANT: You MUST use the provided urgency level exactly as given. Do NOT recalculate or override it.

1. Most likely condition(s). You can mention multiple, but only those that make sense based on the input!
2. Urgency: use the value as provided 
3. Recommendation in 1-2 sentences

Be concise. Always advise seeing a doctor for serious symptoms.
"""

APP_PROMPT = """
Return only valid JSON.
Use exactly this shape:
{
    "conditions": "string",
    "urgency": "LOW | MEDIUM | HIGH | EMERGENCY",
    "recommendation": "string"
}

Do not include markdown fences or extra text. 
Do not include ```json or any other formatting.
Start with { 
End with }
The output must be parseable by json.loads().
"""

PROMPT_CHECKER = """
Does the text describe medical symptoms, injuries, or conditions? Reply "yes" or "no" only.

Examples:
"fever, cough, sore throat" -> yes
"headache, feeling sick" -> yes  
"what is the weather" -> no
"book me a table" -> no
"""

def is_model_available(model: str) -> bool:
    try:
        response = ollama.list()
    except Exception:
        return False

    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])

    for item in models or []:
        installed_name = getattr(item, "model", None)
        if installed_name is None and isinstance(item, dict):
            installed_name = item.get("model") or item.get("name")

        if installed_name == model:
            return True

    return False

def extract_symptoms(model: str, user_input: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )
    return response.message.content.strip()

def get_app_diagnosis(model: str, symptoms: str, rag_response: str, urgency : str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": f"{DIAGNOSE_PROMPT}\n{APP_PROMPT}"},
            {"role": "user", "content": f"Reference context (do NOT assume patient has these):\n{rag_response}"},
            {"role": "assistant", "content": "Understood, I will use this only as background reference."},
            {"role": "user", "content": f"Pre-determined urgency (use exactly): {urgency}\nPatient symptoms: {symptoms}"},
        ],
    )
    return response.message.content.strip()

def prompt_checker(model: str, symptoms: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT_CHECKER},
            {"role": "user", "content": symptoms},
        ],
        options={"temperature": 0} # always pick the option with highest probability (we were getting "no" answer for 
                                   # correct results sometimes
    )
    return response.message.content.strip()
