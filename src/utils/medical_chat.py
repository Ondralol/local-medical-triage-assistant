import ollama

DEFAULT_MODEL = "qwen2.5:0.5b"

EXTRACT_PROMPT = """
Extract only the medically relevant symptoms and injuries from the user's message.
Infer obvious medical implications from the wording.
Treat trauma descriptions as medical injuries.
Return a short comma-separated list of symptoms only. No explanation, no preamble.
"""

DIAGNOSE_PROMPT = """
You are a medical triage assistant. Given a list of symptoms and the RAG response, respond with:
1. Most likely condition(s)
2. Urgency: LOW / MEDIUM / HIGH / EMERGENCY
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
Is this a list of symptoms that a person might have? 
Answer "yes" or "no" in this exact format. No explanation, no preamble.
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

def get_app_diagnosis(model: str, symptoms: str, rag_response: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", 
            "content": f"{DIAGNOSE_PROMPT}{APP_PROMPT}\n\nRelevant medical context:\n{rag_response}"},
            {"role": "user", "content": f"Symptoms: {symptoms}"},
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
    )
    return response.message.content.strip()
