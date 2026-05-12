import argparse
import time
from medical_chat import DEFAULT_MODEL, extract_symptoms, get_diagnosis


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
        diagnosis = get_diagnosis(args.model, symptoms)
        print(diagnosis)

        print(f"\n[{time.time() - t_start:.1f}s total]\n")


if __name__ == "__main__":
    main()
