# Similo

**Similo** is a research project for the computational analysis of **Qur'anic similes** using Arabic Transformer-based language models. The project focuses on automatically identifying the **pragmatic function** (communicative intent) of Qur'anic similes through multi-class text classification.

## Overview

Qur'anic similes are not merely rhetorical devices; they serve diverse pragmatic purposes such as warning, persuasion, clarification, glorification, and consolation. Automatically recognizing these functions remains a challenging task due to the richness of Qur'anic language and the semantic overlap among rhetorical categories.

This project provides a benchmark framework for classifying Qur'anic similes into predefined pragmatic function categories using state-of-the-art Arabic Transformer models.

## Objectives

- Build a benchmark for pragmatic function classification of Qur'anic similes.
- Evaluate modern Arabic Transformer models on this task.
- Analyze the challenges posed by semantic overlap between rhetorical functions.
- Provide a reproducible experimental framework for future research.

## Task Definition

**Input**

- Qur'anic verse
- Simile segment (optionally combined with the verse using a separator token)

**Output**

One of the predefined pragmatic function classes.

## Features

- Arabic Transformer-based classification
- Multi-class text classification
- Support for multiple pretrained models
- Reproducible training and evaluation pipeline
- Error analysis and performance visualization

## Project Structure

```
Similo/
│
├── data/                 # Dataset
├── notebooks/            # Experiments and analysis
├── src/                  # Source code
├── models/               # Saved models
├── results/              # Experimental results
├── figures/              # Figures and visualizations
├── requirements.txt
└── README.md
```

## Models

The framework is designed to evaluate various Arabic pretrained language models, including:

- AraBERT
- CAMeLBERT
- MARBERT

Additional models can be integrated with minimal modifications.

## Evaluation

Typical evaluation metrics include:

- Accuracy
- Macro F1-score
- Precision
- Recall
- Confusion Matrix

## Research Status

This repository accompanies an ongoing academic research project. The implementation, dataset organization, and documentation may evolve as the study progresses.


---

**Similo** — *Computational Analysis of Qur'anic Similes using Arabic Transformer Models.*
