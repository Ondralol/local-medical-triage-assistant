# NLP Project

## Prerequisites
- Ollama (used to manage local LLMs) - Download [here](https://ollama.com/download/)
- Python >= 3.11
- uv - Download [here](https://docs.astral.sh/uv/)

## Setup

### Install local LLM
After successfuly installing Ollama, install local LLM models by using
```
ollama pull {model_name}
```
For tested our project primarily with gemma4:e2b, but you may try with different models

### Run the application
Before running for the first time download dependecies by using
```
uv sync
```
and then run the application using
```
uv run src/main.py --model <model_name>
```
where <model_name> is the name of local LLM you downloaded with ollama

### Run the jupyter notebooks 
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

Note: if using inside sagemaker, copy the pyproject.toml and do the same step. Then select the `NLP Project` kernel.


### Code strucure
- In `/notebooks` are notebooks used for dataset exploration preprocessing
- Main application source code is in the `src/` folder

Have you ever wondered where medical chatbots or intelligent search engines for health information get their knowledge? The answer lies in large datasets like MedQuAD! This rich resource provides a treasure trove of real-world medical questions and informative answers, paving the way for advancements in Natural Language Processing (NLP) and Information Retrieval (IR) within the healthcare domain.

What is MedQuAD?
MedQuAD, short for Medical Question Answering Dataset, is a collection of question-answer pairs meticulously curated from 12 trusted National Institutes of Health (NIH) websites. These websites cover a wide range of health topics, from cancer.gov to GARD (Genetic and Rare Diseases Information Resource).

### Datasets
- To match the diseases and symptoms we used `MedQuAD: Medical Question-Answer Dataset`. Which is a large dataset containing collection of question-answer pairs covering a wide range of medical topics. You can find the dataset [here](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research?resource=download)
- Since the dataset contains a lot of question from different areas and we are mainly interested in the symptopms, we filtered the dataset to only get the questions that contain one of the following: "symptom", "signs", "cause", "disease", "condition", "syndrome"

