# Medical Triage Assistant

We present an application called **Medical Triage Assistant**, created to determine a patient's most likely medical condition, its urgency level, and the recommended next steps based on a free-text description of their symptoms. The pipeline uses local LLMs to extract symptoms from the patient's input, combines Retrieval-Augmented Generation (RAG) and Best Matching 25 (BM25) to retrieve relevant context from a large medical corpus, and applies a fine-tuned BioBERT classifier to determine the medical urgency across three levels: `LOW`, `MEDIUM`, and `HIGH`. All of this information is then processed and presented to the patient in a structured text format.

**The primary goal of the assistant is to provide an initial assessment of a patient that is both private and explainable. Privacy is addressed since everything runs locally, avoiding the cloud-based nature of typical AI medical systems. Explainability is addressed by grounding the assessment in retrieved evidence from a real medical corpus (via RAG and BM25) rather than relying on opaque LLM output alone. The system is intended as a decision-support tool only and does not constitute professional medical advice.**

## Full report: 
View the full report [here](resources/report.pdf)

## Demo

https://github.com/user-attachments/assets/ccb38401-0ae1-4430-9f00-fc368ec9a148

*Showcase of the user interaction (processing time is cut).*

## Pipeline overview

<p align="center">
  <img src="diagrams/vertical_diagram.png" alt="Pipeline diagram" width="750">
</p>

## Detailed information about the pipeline

### 1. Symptom Extraction

User input is passed through a local LLM via **Ollama**, guided by a constrained prompt that extracts only the medically relevant symptoms. For example:

> **Input:** `"I have a headache, a fever and I'm feeling sick and tired"`
> **Extracted:** `"headache, fever, sick, tired"`

### 2. Retrieval: RAG + BM25

The extracted symptoms are used to query the **MedQuAD** dataset, which contains disease names (`focus_area`) and symptom descriptions (`answer`) pairs. Two retrieval strategies run in parallel:

- **RAG** - Symptoms are embedded using [`pritamdeka/S-PubMedBert-MS-MARCO`], a medically fine-tuned sentence transformer, and matched against pre-embedded `answer` fields to retrieve the **top 2** semantically similar records.
- **BM25** - A keyword-based search over the same dataset, requiring no embeddings, returning the **top 5** records by token overlap.

The indices returned by both methods are merged (duplicates removed), giving up to **7** unique retrieved records.

### 3. Urgency Classification

Each retrieved record is filtered down to its disease name (`focus_area`) and the first **175 characters** of its symptom description (`answer`). These filtered records are combined with the originally extracted symptoms and passed to a fine-tuned **BioBERT classifier** (`biobert-base-cased-v1.2`), which predicts one of three urgency levels:

| Level |
|-------|
| `LOW` |
| `MEDIUM`|
| `HIGH` |

### 4. Structured Response Generation

The local LLM is used a final time with the extracted symptoms, the filtered retrieved records, and the BioBERT urgency prediction. It produces a structured output containing:

- **Urgency level**
- **Most likely condition(s)**
- **Recommended next steps**

## Prerequisites
- Ollama (used to manage local LLMs) - Download [here](https://ollama.com/download/)
- Python >= 3.11
- uv - Download [here](https://docs.astral.sh/uv/)

## Setup

### 1. Clone the project

### 2. Download neccessary files for RAG, BM25 and BERT from Huggingface
Copy the contents of the `RAG`, `BM25`, `models` and `dataset` folders from the [link](https://huggingface.co/ondralol/nlp_project/tree/main) to your local project folder. Keep the same structure - meaning that the content of `RAG` folder from HuggingFace needs to go to the local `RAG` folder

Note: for the first run, the programm will download another ~400MB embedding model from HuggingFace, so first load might take while. Other runs should be faster

### 3. Install local LLM
After successfuly installing Ollama, install local LLM models by using
```
ollama pull {model_name}
```
For tested our project primarily with `gemma4:e2b`, `llama3.2:3b `qwen2.5:0.5b`, but you may try with different models but making simple changes

### 4. Download all the dependencies
Before running for the first time download dependecies by using
```
uv sync
```

### 5. Run the application
You can run the application using
```
uv run src/main.py
```


## Running the jupyter notebook
First run
```
uv sync
```
and in order to install kernel run 
```
uv run python -m ipykernel install --user --name=nlp-project --display-name "NLP Project"
```
Finally to open the notebook run
```
uv run jupyter notebook
```
and open the appropriate notebook

Note: if using inside sagemaker, copy the pyproject.toml and do the same steps. Then select the `NLP Project` kernel.


## Project strucure
- In `/notebooks` are notebooks used for dataset exploration preprocessing and model training
- Main application source code is in the `src/` folder