# Similo: A Computational Framework for the Pragmatic Analysis of Qur'anic Similes

**Similo** is an interdisciplinary research project situated at the intersection of **Classical Arabic Rhetoric (*Balāgha*)**, **Qur'anic Studies**, and **Artificial Intelligence (NLP)**. The project focuses on the computational analysis of Qur'anic similes (*Tashbīh*), aiming to automatically identify their underlying **pragmatic functions** and communicative intents (*Al-Aghrāḍ al-Balāghiyyah* / *Al-Maqāṣid*).

## 📖 Overview

In the rhetorical ecology of the Qur'an, a simile (*Tashbīh*) is rarely used as a mere aesthetic ornament (*zaḥrafa lafẓiyyah*). Instead, it functions as a dynamic communicative act designed to influence the receiver. Depending on the discourse context (*Siyāq*), a simile may serve deep pragmatic purposes, such as:

*   **Warning & Intimidation** (*Inthār / Takhwīf*)
*   **Condemnation & Criticism** (*Tawbīkh / Dhamm*)
*   **Argumentation & Persuasion** (*Iqnā' / Ḥujjiyyah*)
*   **Consolation & Reassurance** (*Tasliyah / Muwāsāh*)
*   **Clarification & Imagery** (*Tawḍīḥ / Taqrīb al-Ma'nā*)

Traditionally, uncovering these underlying intents has been the exclusive domain of human exegetes (*Mufassirīn*) and rhetoricians. **Similo** bridges this hermeneutic gap by formalizing Qur'anic pragmatic analysis as a machine-readable task, utilizing state-of-the-art context-aware Neural Networks (Transformer models) to map linguistic structure to rhetorical intent.

## 🎯 Research Objectives

*   **Computational Balāgha:** To build the first standardized computational benchmark for classifying the illocutionary force (pragmatic intent) of Qur'anic similes.
*   **Pre-training Ecologies Evaluation:** To investigate whether modern AI models—trained on Modern Standard Arabic (MSA) news or dialectal social media discourse—can truly comprehend the archaic lexicon and deep rhetorical structures of Classical Arabic.
*   **Modeling Rhetorical Polyvalence (*Iḥtimāl*):** To computationally analyze the semantic overlap where a single sacred verse simultaneously embodies multiple intents (e.g., condemning a past action while warning of future consequences).
*   **Digital Humanities Infrastructure:** To provide a reproducible, open-source pipeline that empowers linguists and Qur'anic scholars to integrate deep learning into traditional textual analysis.

## 🧠 The Conceptual Workflow

Unlike traditional keyword-matching algorithms, **Similo** employs deep contextual representation learning. The framework operates through the following linguistic-computational pipeline:

1.  **The Input Text:** The complete Qur'anic verse (*Āyah*) along with the specific simile segment (*Tashbīh*), preserving the structural context.
2.  **Contextual Embedding:** Deep learning encoders read the text bidirectionally to capture subtle morphological, syntactic, and semantic cues (*Qarā'in*).
3.  **The Output (Pragmatic Decision):** The model calculates a probability distribution across eight predefined communicative functions and predicts the dominant rhetorical intent.

## 🤖 Evaluated Linguistic Models

To determine which linguistic exposure best equips AI to understand Qur'anic rhetoric, the framework tests distinct Transformer ecologies:
*   **Classical Arabic-Specific:** Models trained on historical texts (e.g., CAMeLBERT-CA).
*   **Formal MSA-Centric:** Models trained on modern news and literature (e.g., AraBERT, ARBERT).
*   **Dialect & Social Media-Centric:** Models trained on highly emotive, intent-driven human interactions (e.g., MARBERT).
*   **Multilingual Baselines:** Cross-lingual models covering 100+ languages (e.g., XLM-RoBERTa).

## 💡 Why Similo Matters for Linguists & Exegetes?

**Similo** is not just an engineering pipeline; it is a tool for **Digital Hermeneutics**. By observing how AI struggles to separate closely related rhetorical intents (like *Warning* vs. *Condemnation*), scholars can quantitatively validate the inherent polyvalence and layered meanings (*Ta'addud al-Ma'ānī*) that classical scholars have debated for centuries. It opens new frontiers for studying the Qur'an beyond manual annotation.

## 📌 Research Status & Reproducibility

This repository accompanies an ongoing academic research project aimed at advancing **Arabic Digital Humanities**. The repository contains the source code, dataset structures, and evaluation scripts required to reproduce the study's empirical findings. 

As the research progresses, further annotations and multi-label classification frameworks will be integrated to better capture the multi-functional nature of Qur'anic discourse.

---

**Similo** — *Mapping the Rhetoric of the Qur'an through the Lens of Artificial Intelligence.*
