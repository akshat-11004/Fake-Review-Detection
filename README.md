# Fake Review Detection

## 1. Overview
This notebook focuses on detecting fake reviews using machine learning & NLP. It includes data preprocessing, text cleaning, and three machine learning models:
- **Naïve Bayes Classifier**
- **Support Vector Machine (SVM)**
- **Logistic Regression**

## 2. Methods Used

### **Data Preprocessing & Text Cleaning**
The notebook defines multiple functions for text processing:
- `clean_text(text)`: Cleans text by removing special characters, punctuation, and numbers.
- `preprocess(text)`: Applies text normalization steps.
- `stem_words(text)`: Uses stemming to reduce words to their root form.
- `lemmatize_words(text)`: Uses lemmatization to convert words into base form.
- `reviewsprocess(review)`: Combines all preprocessing steps.

### **Machine Learning Models**
The notebook trains and evaluates **three classifiers**:
1. **Naïve Bayes Classifier** – A probabilistic approach for text classification.
2. **Support Vector Machine (SVM)** – A robust classifier for separating fake and real reviews.
3. **Logistic Regression** – A statistical model used for binary classification.

## 3. Model Training
- Training on labeled data.
- Evaluating models on test sets.
- Comparing accuracy, precision, recall, and F1-score.

## 4. Summary
My notebook follows a structured pipeline:
1. **Data loading and cleaning** (text preprocessing).
2. **Feature extraction** (possibly TF-IDF or CountVectorizer).
3. **Training three different classifiers** (Naïve Bayes, SVM, Logistic Regression).
4. **Evaluating performance** using standard ML metrics.

## Output of the Classification Report:
```plaintext
Classification Report for Naïve Bayes Classifier:
              precision    recall  f1-score   support

          CG       0.82      0.89      0.85      7011
          OR       0.88      0.81      0.84      7136

    accuracy                           0.85     14147
   macro avg       0.85      0.85      0.85     14147
weighted avg       0.85      0.85      0.85     14147


Classification Report for SVM:

              precision    recall  f1-score   support

          CG       0.90      0.86      0.88      7011
          OR       0.87      0.90      0.88      7136

    accuracy                           0.88     14147
   macro avg       0.88      0.88      0.88     14147
weighted avg       0.88      0.88      0.88     14147


Classification Report for Logistic Regression:

              precision    recall  f1-score   support

          CG       0.87      0.85      0.86      7011
          OR       0.86      0.88      0.87      7136

    accuracy                           0.87     14147
   macro avg       0.87      0.87      0.87     14147
weighted avg       0.87      0.87      0.87     14147

```

