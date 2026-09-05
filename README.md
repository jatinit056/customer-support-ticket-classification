# Customer Support Ticket Classification

A notebook-first machine-learning project for cleaning, exploring, training, and evaluating customer-support ticket classifiers with Python and Scikit-learn.

## What the notebook covers

- Dataset loading and inspection
- Missing-value, duplicate, and text-quality checks
- Ticket category, priority, channel, and text-length analysis
- Leakage-safe TF-IDF feature preparation
- Stratified train/test split
- Logistic Regression, LinearSVC, and Multinomial Naive Bayes comparison
- Accuracy, precision, recall, macro/weighted F1, classification report, and confusion matrix
- Influential terms, error analysis, sample predictions, limitations, and conclusion

## Dataset

The included `synthetic_support_tickets.csv` contains 2,100 reproducibly generated development tickets across seven balanced categories. It contains no real customer information and must not be represented as real-world data.

The preferred real dataset is [Suraj520's Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset), which Kaggle lists as CC0/Public Domain. Replace the synthetic CSV and update the loading/mapping cell before making real-world performance claims.

## Measured results

On the included synthetic 420-record holdout, Logistic Regression, LinearSVC, and Multinomial Naive Bayes each achieved 1.000 accuracy and macro/weighted F1. This perfect result reflects easily separable generated templates and validates the workflow—not production generalization.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook customer_support_ticket_classification.ipynb
```

Then choose **Kernel → Restart & Run All**.

## Files

- `customer_support_ticket_classification.ipynb` — complete reproducible workflow
- `synthetic_support_tickets.csv` — clearly labeled development dataset
- `requirements.txt` — notebook dependencies
- `LICENSE` — project license

## Author

Jatin Gohel — [GitHub](https://github.com/jatinit056)
