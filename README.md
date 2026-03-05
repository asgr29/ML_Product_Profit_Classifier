## ML Product Profit Classification

## Description of the Problem

Retail companies often manage large catalogs of products with varying levels of profitability. Some products consistently generate profit, while others can lead to financial losses due to factors such as discounts, shipping costs, or operational expenses.

Identifying products that are likely to generate losses is valuable for businesses because it allows them to:

* Adjust pricing strategies
* Optimize product portfolios
* Improve operational decisions
* Reduce financial risk

The objective of this project is to build a machine learning classification model capable of predicting whether a product will generate profit or loss, enabling more informed business decisions.

## Dataset Used

The dataset used in this project is the Superstore dataset, a publicly available dataset commonly used for analytics and machine learning exercises.

It contains transactional information about retail orders, including:

* Sales
* Profit
* Discount
* Shipping details
* Product categories
* Customer and regional information

The dataset is included in this repository for reproducibility.

Location in repository:

src/data_sample/Superstore.csv


## Solution Adopted

The business problem was framed as a binary classification problem.

A binary target variable was created from the original Profit column:

1 → Product generates profit
0 → Product generates loss

The machine learning workflow followed these steps:

1. Data loading and preparation
2. Creation of the binary target variable
3. Train-test split with stratification
4. Feature preprocessing
5. Handling categorical variables
6. Feature scaling and normalization
7. Model training and evaluation
8. Hyperparameter tuning using GridSearchCV
9. Selection of the best model based on Recall

The models evaluated include:

* Logistic Regression
* Decision Tree
* Random Forest
* LightGBM
* XGBoost

Recall was selected as the primary metric because detecting products that generate losses is more important than maximizing overall accuracy.

The final trained model is stored in the repository for reuse.

## Repository Structure
ML_PRODUCT_PROFIT_CLASSIFICATION

src/
│
├── data_sample/
│   └── Superstore.csv
│
├── img/
│   └── img.jpg
│
├── models/
│   ├── logistic_regression.joblib
│   └── model.pkl
│
├── notebooks/
│   ├── notebook.ipynb
│   └── notebook_sucio.ipynb
│
├── utils/
│   ├── bootcampviztools.py
│   └── toolbox_ML.py
│
├── main.ipynb
├── presentacion.pdf
├── README.md
└── .gitignore

#### Folder description

data_sample/ → Dataset used for the project

img/ → Images used in the presentation or analysis

models/ → Saved trained models

notebooks/ → Development notebooks used during experimentation

utils/ → Helper functions and visualization tools

main.ipynb → Main notebook containing the final pipeline

presentacion.pdf → Project presentation


## Technologies Used

The project was developed using the following technologies:

##### Programming Language

Python

##### Libraries

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* joblib

##### Tools

* Jupyter Notebook
* Git
* GitHub
* VS Code

##### Custom Utility Modules

The project also includes custom helper modules located in the utils/ directory:

bootcampviztools.py — visualization utilities used during exploratory data analysis (EDA), including functions for plotting categorical distributions, relationships between variables, and exploratory graphs. 

toolbox_ML.py — machine learning helper functions for dataset inspection, variable typing, statistical testing, and feature selection during the modeling process.


## Instructions to Reproduce the Project
1. Clone the repository
git clone https://github.com/yourusername/ML_PRODUCT_PROFIT_CLASSIFICATION.git

2. Navigate to the project directory
cd ML_PRODUCT_PROFIT_CLASSIFICATION

3. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

4. Run the main notebook

Open the notebook:
    main.ipynb
and execute all cells to reproduce the full machine learning pipeline.

## Main Results

Several machine learning models were evaluated to determine the best approach for predicting whether a product will generate profit or loss. The models tested included:

* Logistic Regression
* Decision Tree
* Random Forest
* LightGBM
* XGBoost

Each model was trained and evaluated using the same train-test split and preprocessing pipeline to ensure a fair comparison.

Because the business objective is to identify products that may generate losses, the primary evaluation metric used was Recall. In this context, recall measures the model’s ability to correctly detect loss-generating products, minimizing the risk of failing to identify problematic items.

Hyperparameter tuning was performed using GridSearchCV to optimize model performance.

After evaluating all models, Logistic Regression was selected as the final model. Despite the availability of more complex models, Logistic Regression provided the best balance between recall performance, model stability, and interpretability.

The final optimized model achieved strong recall performance on the test set, meaning it successfully identifies a large proportion of products that generate losses. This makes the model particularly useful for supporting business decisions such as pricing adjustments, product portfolio optimization, and risk reduction.

The trained model is saved in the repository and can be reused for future predictions.

## Authors

Nick Brown
GitHub: https://github.com/Nick-Brown-Git
LinkedIn: https://linkedin.com/in/nick-brown-bb218a3a3/
Ana Sofía Gómez Ramírez
Github: https://github.com/asgr29