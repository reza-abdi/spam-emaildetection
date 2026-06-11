# Spam Detection with Logistic Regression

A machine learning project that classifies emails as spam or ham (not spam) using natural language processing and logistic regression.

## Overview

This notebook walks through a complete spam detection pipeline — from raw text data to a trained classifier — using a bag-of-words approach with scikit-learn.

## Dataset

The model is trained on `spam_dataset.csv`, which contains two columns:

| Column | Description |
|--------|-------------|
| `text`  | The raw email/message content |
| `label` | Classification label (`spam` or `ham`) |

## Workflow

1. **Data Loading & Exploration** — Load the CSV, preview rows, and inspect basic statistics
2. **Data Cleaning** — Drop rows with missing values
3. **Feature Extraction** — Convert raw text into numeric feature vectors using `CountVectorizer` (bag-of-words)
4. **Train/Test Split** — Split data 80/20 with a fixed random seed (`random_state=42`)
5. **Model Training** — Fit a Logistic Regression classifier (`max_iter=1000`)
6. **Evaluation** — Measure accuracy, generate a confusion matrix, and print a full classification report
7. **Inference** — Predict the label of a new, unseen email

## Requirements

```
pandas
scikit-learn
matplotlib
```

Install dependencies with:

```bash
pip install pandas scikit-learn matplotlib
```

## Usage

1. Place `spam_dataset.csv` in the same directory as the notebook
2. Open and run `spamDetection.ipynb` top to bottom

To classify a new message, update the `new_email` variable in the final cell:

```python
new_email = "Your message here"
new_email_vectorized = vectorizer.transform([new_email])
prediction = model.predict(new_email_vectorized)
print("Prediction:", prediction)
```

## Example Prediction

```
Input:  "Congratulations you have won one billion dollars for just breathing in..."
Output: ['spam']
```

## Model Performance

Evaluated on a held-out test set of 1,035 messages (80/20 split).

**Accuracy: 98.1%**

**Confusion Matrix:**

|               | Predicted Ham | Predicted Spam |
|---------------|:---:|:---:|
| **Actual Ham**  | 732 | 10 |
| **Actual Spam** | 10  | 283 |

**Classification Report:**

| Class    | Precision | Recall | F1-Score | Support |
|----------|:---------:|:------:|:--------:|:-------:|
| Ham      | 0.99      | 0.99   | 0.99     | 742     |
| Spam     | 0.97      | 0.97   | 0.97     | 293     |
| **Weighted Avg** | **0.98** | **0.98** | **0.98** | **1035** |

The model misclassified 20 messages in total — 10 false positives (ham flagged as spam) and 10 false negatives (spam that slipped through).
