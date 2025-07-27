# Student Explanation Classification

This repository contains the code and data for a machine learning project focused on classifying student explanations in mathematics.

## Project Structure

```
map-charting-student-math-misunderstandings/
├── student_explanation_classification.ipynb  # Main Jupyter notebook with analysis
├── train.csv                                 # Training dataset
├── test.csv                                  # Test dataset
├── sample_submission.csv                     # Sample submission format
└── label_mapping.json                        # Label mapping for classifications
```

## Important Note

The following large model files are excluded from this repository due to GitHub's file size limits (100MB):

- `best_student_explanation_model.pkl` (209.34 MB)
- `best_model.pkl`
- `label_encoder.pkl`
- `tfidf_vectorizer.pkl`
- `vectorizer.pkl`

These files contain trained machine learning models and vectorizers that are generated during the model training process. They are excluded via the `.gitignore` file to prevent repository size issues.

## Getting Started

1. Clone this repository
2. Run the Jupyter notebook `student_explanation_classification.ipynb`
3. The notebook will train the models and generate the necessary `.pkl` files locally

## Requirements

- Python 3.x
- Jupyter Notebook
- Required Python packages (see notebook for imports)

## Model Files

When you run the notebook, it will generate the following model files locally:
- `best_student_explanation_model.pkl` - Trained classification model
- `best_model.pkl` - Best performing model
- `label_encoder.pkl` - Label encoder for target variables
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer for text features
- `vectorizer.pkl` - General vectorizer

These files will be created in the same directory as the notebook when you execute the training cells. 