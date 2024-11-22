# Text Analysis and classification for Information Retrieval

# README

This repository contains a personal project focused on three main tasks:

1. **IR Evaluation**: A module to evaluate Information Retrieval (IR) systems using various retrieval scores.
2. **Text Analysis**: Analysis of text from the Quran and Bible using Mutual Information (MI), Chi-Square (χ2), and Latent Dirichlet Allocation (LDA).
3. **Text Classification**: Implementation of a sentiment analyzer for three-way sentiment classification of tweets.

## IR Evaluation
- **Objective**: Evaluate IR systems using different retrieval scores.
- **Inputs**: `system_results.csv` and `qrels.csv`.
- **Measures**: P@10, R@50, r-precision, AP, nDCG@10, nDCG@20.
- **Output**: `ir_eval.csv` containing evaluation results for 6 systems.

## Text Analysis
- **Objective**: Analyze verses from the Quran, New Testament, and Old Testament.
- **Tasks**:
    - Preprocess data.
    - Compute MI and χ2 scores for tokens.
    - Run LDA with k=20 topics.
- **Output**: Ranked lists of tokens and topic analysis.

## Text Classification
- **Objective**: Implement a sentiment analyzer for tweets.
- **Tasks**:
    - Split dataset into training and development sets.
    - Train baseline SVM classifier with BOW features.
    - Improve classifier based on error analysis.
- **Output**: `classification.csv` containing classification results for baseline and improved models.
