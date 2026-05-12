import argparse
import time
import ollama

DEFAULT_MODEL = "qwen2.5:0.5b" 

EXTRACT_PROMPT = """Extract only the medical symptoms from the user's message.
Return a short comma-separated list of symptoms only. No explanation, no preamble."""

DIAGNOSE_PROMPT = """You are a medical triage assistant. Given a list of symptoms respond with:
1. Most likely condition(s)
2. Urgency: LOW / MEDIUM / HIGH / EMERGENCY
3. Recommendation in 1-2 sentences

Be concise. Always advise seeing a doctor for serious symptoms."""


def extract_symptoms(model: str, user_input: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )
    return response.message.content.strip()


def stream_diagnosis(model: str, symptoms: str):
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": DIAGNOSE_PROMPT},
            {"role": "user", "content": f"Symptoms: {symptoms}"},
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk.message.content, end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Symptom  CLI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    print(f"Symptom check using {args.model}")
    print("Describe how you feel. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        t_start = time.time()

        print("Extracting symptoms...", end="\r", flush=True)
        symptoms = extract_symptoms(args.model, user_input)
        print(f"Symptoms: {symptoms}\n")

        print("Assessment:\n")
        stream_diagnosis(args.model, symptoms)

        print(f"\n[{time.time() - t_start:.1f}s total]\n")


if __name__ == "__main__":
    main()
