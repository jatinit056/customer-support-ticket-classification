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

Logistic Regression was selected after comparing it with LinearSVC and Multinomial Naive Bayes. On the included 420-record synthetic holdout it achieved **84.3% accuracy**, **0.843 macro F1**, and **0.843 weighted F1**. This approximately 85% result is measured and reproducible, but it remains a synthetic benchmark—not production validation.

## Resume highlights

- Cleaned and prepared 2,100 synthetic customer-support tickets for classification and operational analysis.
- Performed exploratory analysis to identify ticket-category, priority, channel, resolution-time, and satisfaction patterns.
- Compared three machine-learning classifiers; the selected Logistic Regression pipeline achieved 84.3% accuracy and 0.843 macro F1 on a 420-record synthetic holdout.
- Created Matplotlib visualizations and documented the complete reproducible workflow in a self-contained Jupyter Notebook.

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
