## Yelp Review Sentiment Classification using DistilBERT

This project fine tunes a DistilBERT transformer model to classify Yelp reviews into five rating-based sentiment classes using the `yelp_review_full` dataset from Hugging Face. The notebook performs dataset loading, exploratory data analysis, model training, evaluation and sentiment prediction on custom restaurant review examples.

## Project Overview

The main goal of this project is to build a text classification model that can understand customer reviews and predict their sentiment/rating category. The model is trained on Yelp review text and uses five labels representing review ratings from 0 to 4.

After training, the notebook also maps the predicted rating labels into broader sentiment categories:

- Labels `0` and `1`: Negative
- Label `2`: Neutral
- Labels `3` and `4`: Positive

## What This Notebook Does

1. Installs and imports the required libraries.
2. Loads the `yelp_review_full` dataset from Hugging Face.
3. Converts the training and test splits into pandas DataFrames.
4. Checks the dataset structure, shape, and missing values.
5. Performs exploratory data analysis, including:
   - Label distribution
   - Review length distribution
   - Class balance check
   - Word clouds for positive and negative reviews
6. Loads the `distilbert-base-uncased` tokenizer and sequence classification model.
7. Samples a smaller subset of the dataset for faster experimentation:
   - 20,000 training samples
   - 5,000 validation samples
8. Tokenizes review text with truncation, padding, and a maximum length of 128 tokens.
9. Creates a custom PyTorch Dataset class for model training.
10. Fine-tunes DistilBERT using Hugging Face `Trainer`.
11. Evaluates the model using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
12. Defines a `classify_review()` function for sentiment prediction.
13. Tests the trained model on custom restaurant review examples.

## Dataset

The project uses the Hugging Face dataset:

```python
load_dataset("yelp_review_full")
```

The dataset contains Yelp reviews with labels from `0` to `4`, where each label represents a review rating category.

Dataset columns:

- `text`: Review text
- `label`: Rating/sentiment class

The full training dataset contains 650,000 rows, but the notebook samples 20,000 rows for training and 5,000 rows for validation to reduce training time.

## Technologies Used

- Python
- Google Colab
- Hugging Face Datasets
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud

## Model

The notebook uses:

```python
DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=5
)
```

DistilBERT is a smaller and faster version of BERT. It is suitable for text classification tasks while requiring fewer computational resources than full BERT.

## Training Configuration

The model is trained using Hugging Face `Trainer` with the following settings:

```python
TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    fp16=True
)
```

Mixed precision training (`fp16=True`) is enabled, which is useful when running the notebook on a compatible GPU.

## Evaluation

The notebook evaluates the trained model using:

- Classification report
- Precision per class
- Recall per class
- F1-score per class
- Confusion matrix

The evaluation code predicts labels on the validation dataset and compares them with the true labels.

## Sentiment Prediction Function

The notebook includes a function called `classify_review()` that accepts a review as input and returns one of three sentiment categories:

```python
Negative
Neutral
Positive
```

Example:

```python
classify_review("The food was delicious, and the service was excellent!")
```

Expected output:

```python
Positive
```

## Conclusion

This project demonstrates how to fine tune a transformer based language model for sentiment classification using Yelp review data. It covers the complete machine learning workflow, including data loading, EDA, preprocessing, model training, evaluation and prediction on new text samples.
