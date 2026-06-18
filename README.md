# SMS Spam Detection Pipeline

An end-to-end Machine Learning project designed to classify SMS messages as spam or legitimate (ham) using Natural Language Processing techniques and optimized machine learning models.

This repository covers the complete NLP workflow, including Exploratory Data Analysis (EDA), text preprocessing, TF-IDF feature extraction, model benchmarking, hyperparameter tuning, probability calibration, and final model deployment.

---

# Project Architecture & Workflow

The project is divided into two main components to ensure a clean and reproducible machine learning workflow:

1. **Exploratory Data Analysis (`eda.ipynb`)**
   - Dataset investigation
   - Data cleaning
   - Text preprocessing
   - Class distribution analysis
   - Text feature analysis
   - Word frequency analysis
   - WordCloud visualization


2. **Model Training Pipeline (`training.ipynb`)**
   - TF-IDF feature extraction
   - Machine learning model comparison
   - Cross-validation evaluation
   - Hyperparameter tuning
   - Final model training
   - Probability prediction
   - Model serialization

---

# Dataset

The dataset used in this project was obtained from Kaggle:

**Dataset Source:**  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset


The dataset contains SMS messages classified into two categories:

- `ham` → legitimate messages
- `spam` → unwanted messages


Dataset features:

| Feature | Description |
|---|---|
| label | Message category |
| text | SMS message content |


The original dataset file used in this project:

```
spam.csv
```

> Dataset license information was not specified by the original source. Please refer to the Kaggle dataset page for usage terms.

---

# 1. Exploratory Data Analysis (EDA)

During the EDA phase, the raw SMS data was analyzed and prepared for machine learning.

---

# Data Cleaning & Preparation

The following preprocessing steps were applied:

- Removed unnecessary columns
- Renamed columns for clarity
- Converted text to lowercase
- Removed URLs
- Removed extra whitespace
- Removed duplicate messages


Example:

Before:

```
FREE Prize!!! Visit http://example.com
```


After:

```
free prize!!!
```

---

# Missing Value Analysis

The dataset was checked for missing values.

Result:

```
No missing values were found.
```

---

# Class Distribution Analysis

The target variable:

```
label
```

was analyzed.

The dataset contains an imbalanced distribution between:

- Ham messages
- Spam messages


Because of this imbalance, accuracy alone was not considered a reliable evaluation metric.

The main evaluation focus was:

- Precision
- Recall
- F1-Score

for the spam class.

---

# Text Feature Analysis

Additional features were created for exploratory analysis:

## Message Length

The number of characters in each message.


## Word Count

The number of words in each message.


## Special Character Count

The number of punctuation and special characters in each message.


These features were analyzed to identify differences between spam and ham messages.

---

# Word Frequency Analysis

The most frequent words were extracted separately for:

- Spam messages
- Ham messages


Visualization:

- Word frequency counter
- WordCloud


---

# 2. Text Feature Extraction

Machine learning models cannot directly process raw text.

Therefore, TF-IDF Vectorization was used to convert text messages into numerical features.

Pipeline:

```
Raw Text
    |
    ↓
TF-IDF Vectorizer
    |
    ↓
Machine Learning Model
```

---

# 3. Model Training & Evaluation

Multiple machine learning models were trained and compared.

Models evaluated:

| Model |
|---|
| Multinomial Naive Bayes |
| Logistic Regression |
| Support Vector Machine (SVC) |
| XGBoost Classifier |


Model comparison was performed using:

```
Stratified K-Fold Cross Validation
```

---

# Evaluation Metrics

Since the dataset is imbalanced, the main evaluation metrics were focused on the spam class.

---

## Precision (Spam)

Measures how many messages predicted as spam were actually spam.

A higher precision reduces false spam predictions.

---

## Recall (Spam)

Measures how many actual spam messages were successfully detected.

---

## F1-Score

The main metric used for model selection.

F1-score provides a balance between precision and recall.

---

# 4. Hyperparameter Tuning

The best performing model from baseline evaluation was:

```
Support Vector Machine (SVC)
```


Hyperparameter optimization was performed using:

```
GridSearchCV
```


Parameters tuned:

- Kernel
- C
- Gamma
- TF-IDF max features


---

# Final Model

The final model pipeline:

```
TF-IDF Vectorizer
        |
        ↓
       SVC
        |
        ↓
Probability Calibration
```


Probability calibration was applied using:

```
CalibratedClassifierCV
```


This allows the model to return prediction confidence.

Example output:

```python
{
    "prediction": "Spam",
    "probability": "98.40%"
}
```

---

# Example Predictions


Input:

```
FREE prize click now!!!
```


Output:

```
Prediction: Spam
Probability: 98%
```


---

Input:

```
Hey, are we still meeting for lunch tomorrow?
```


Output:

```
Prediction: Ham
Probability: 99%
```

---

# Project Structure

```
spam_detection_project/

│
├── data/
│   ├── spam.csv
│   └── spam_processed.csv
│
├── source_code/
│   ├── eda.ipynb
│   └── training.ipynb
│
├── final_model/
│   └── spam_detector_model.pkl
│
├── README.md
│
└── requirements.txt
```

---

# How to Run & Reproduce


## 1. Clone Repository

```bash
git clone https://github.com/mohammad-javad-0/SMS-Spam-Detection.git

cd spam_detection_project
```


---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```


---

## 3. Dataset Setup

Download the dataset from Kaggle:

```
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
```


Place the dataset file:

```
spam.csv
```

inside:

```
data/spam.csv
```


---

## 4. Execute Notebooks

Run:

```
source_code/eda.ipynb
```

for data analysis and preprocessing.


Then run:

```
source_code/training.ipynb
```

for model training, tuning, evaluation, and saving the final model.

---

# Future Improvements

Possible improvements:

- Deep learning models (LSTM / Transformers)
- Advanced text normalization
- Hyperparameter optimization using Optuna
- Deployment using FastAPI or Streamlit

---

# License & Dataset Attribution

This project uses a dataset obtained from Kaggle.

Dataset source:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset


Please refer to the original dataset page for dataset license and usage terms.

---
# Author

Javad Mahdavi

Machine Learning & NLP Projects