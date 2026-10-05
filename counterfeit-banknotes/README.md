# Spotting counterfeit banknotes from six measurements

Comparing four machine-learning models to tell genuine euro banknotes from fakes using their geometry alone, then shipping the best one as a ready-to-use prediction script.

**Full case study:** [axeldrtdata.github.io/works/counterfeit-banknotes](https://axeldrtdata.github.io/works/counterfeit-banknotes/)

## Context

The ONCFM, an anti-counterfeiting agency (fictional OpenClassrooms case), records six measurements for every banknote it receives: length, diagonal, height on each side, and the upper and lower margins. The goal was to build an algorithm that classifies a note as genuine or fake from these measurements only, and to catch as many counterfeits as possible.

## Results

| Model                 | Learning     | Accuracy | Errors / 300 | AUC    |
| --------------------- | ------------ | -------- | ------------ | ------ |
| Logistic regression   | Supervised   | 99.0%    | 3            | 0.9995 |
| Random forest         | Supervised   | 99.0%    | 3            | 0.9994 |
| K-nearest neighbours  | Supervised   | 98.7%    | 4            | 0.9972 |
| K-means (2 clusters)  | Unsupervised | 98.7%    | 4            | —      |

**Logistic regression was selected**: same performance as the random forest, but simpler, faster and explainable. On the 300-note test set, it caught 98 of the 100 counterfeits.

## Repository structure

```
counterfeit-banknotes/
├── notebooks/
│   ├── 01-analysis-and-modelling.ipynb   # EDA, preparation, 4 models compared
│   └── 02-prediction-script.ipynb        # loads the model and scores new notes
├── data/
│   ├── billets.csv                       # 1,500 labelled notes (training data)
│   └── billets_production.csv            # sample of new notes to score
├── model/
│   └── counterfeit_detection_model.joblib
├── requirements.txt
└── README.md
```

## How to run it

```bash
git clone https://github.com/axeldrtdata/counterfeit-banknotes.git
cd counterfeit-banknotes
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/02-prediction-script.ipynb` to score new banknotes with the trained model. Each note gets a verdict, a probability of being genuine, and a "To check" flag when the model is uncertain.

## Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Plotly · Jupyter

## Note

The notebooks are written in French, as they were produced during my OpenClassrooms Data Analyst training. The case study linked above presents the full project in English.

---

Axel Derobert · Data Analyst · [Portfolio](https://axeldrtdata.github.io) · [LinkedIn](https://www.linkedin.com/in/axel-derobert-5717463b1/)
