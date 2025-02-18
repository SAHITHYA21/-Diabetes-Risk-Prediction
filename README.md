# A Predictive Assessment System for Diabetes Risk Factors

## Overview
This project aims to analyze various health-related factors to predict diabetes risk using machine learning models implemented in PySpark’s MLlib. The dataset was sourced from Kaggle and underwent extensive preprocessing, feature engineering, and modeling to ensure high accuracy and reliable predictions.

## Dataset
The dataset consists of 70,707 rows and 19 columns, with both numerical and categorical features:

- **Numerical Features**: Age, HighChol, CholCheck, BMI, HeartDiseaseorAttack, PhysActivity, HvyAlcoholConsump, MentHlth, PhysHlth, HighBP.
- **Categorical Features**: Sex, Smoker, Fruits, Veggies, GenHlth, DiffWalk, Stroke, SugarConsumption, Diabetes.

## Data Preprocessing
The dataset was cleaned and prepared through the following steps:
1. **Removing Duplicate Rows**: 5,554 duplicate entries removed.
2. **Handling Missing Values**: Missing values were removed or replaced using mode (categorical) and mean (numerical).
3. **Fixing Inconsistencies**: Standardizing categorical labels (e.g., 'Male' and 'M' were unified as 'M').
4. **Removing Negative Values**: 11 negative age values removed.
5. **Feature Engineering**:
   - Created `Age_Group` (Kid, Teen)
   - Created `BMI_Group` (Underweight, Healthy, Overweight, Obese)
6. **Label Encoding**: Categorical values converted into numerical labels.
7. **Outlier Removal**: Used IQR method to remove extreme values.
8. **Rolling Average BMI**: Computed a smoothed BMI value over a rolling window.

## Machine Learning Models
Several machine learning models were trained and evaluated:
1. **Logistic Regression**
2. **Naïve Bayes**
3. **Support Vector Machines (SVM)**
4. **Random Forest**
5. **Gradient Boosting (XGBoost)**
6. **Decision Tree**
7. **Multilayer Perceptron (MLP)**

## Performance Comparison
- **Accuracy**:
  - Logistic Regression, SVM, Random Forest, Gradient Boost, and MLP achieved ~78-79% accuracy.
  - Naïve Bayes and Decision Tree performed lower (~66-71%).
- **Execution Time**:
  - Naïve Bayes and Decision Tree were fastest (~3-4 sec).
  - Logistic Regression and XGBoost were moderate (~11-12 sec).
  - MLP was slowest (~16 sec).

## Visualization & Analysis
- DAG visualizations were used to track Spark job execution stages.
- Confusion matrices were analyzed for each model.
- F1-score, precision, and recall were calculated for evaluation.

## Installation & Usage
### Prerequisites
- Python 3.8+
- Apache Spark & PySpark
- Jupyter Notebook
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `pyspark`, `sklearn`

### Running the Project
```bash
# Clone the repository
git clone https://github.com/yourusername/diabetes-risk-prediction.git
cd diabetes-risk-prediction

# Set up virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run PySpark processing script
spark-submit scripts/preprocessing.py

# Train models
spark-submit scripts/train_models.py
```
