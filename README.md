# House Price Prediction using Machine Learning

A comprehensive machine learning project that predicts house prices using the Ames Housing dataset. This project implements a complete data science pipeline from data preprocessing to model evaluation and feature importance analysis.

## 📊 Project Overview

This project analyzes housing data to predict sale prices using various machine learning algorithms. The pipeline includes data cleaning, exploratory data analysis, feature engineering, and model comparison to identify the best performing approach for house price prediction.

## 🎯 Objective

Build an accurate and robust machine learning model to predict house prices based on various property features, providing insights into the most important factors that influence housing prices.

## 📁 Dataset

- **Dataset**: Ames Housing Dataset (`AmesHousing.csv`)
- **Records**: 2,930 houses
- **Features**: 82 property characteristics
- **Target Variable**: SalePrice (transformed to log scale)

## 🛠️ Technologies Used

- **Python 3.12.6**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `matplotlib` & `seaborn` - Data visualization
  - `scikit-learn` - Machine learning algorithms
  - `xgboost` - Gradient boosting framework

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## 📈 Project Structure

### Step 1: Data Loading
- Import the Ames Housing dataset
- Initial data exploration and shape analysis

### Step 2: Data Preprocessing
- Data quality assessment
- Missing value analysis
- Duplicate detection
- Outlier identification using IQR method

### Step 3: Target Variable Analysis
- Distribution analysis of SalePrice
- Skewness and kurtosis evaluation
- Log transformation to normalize the target variable

### Step 4: Exploratory Data Analysis (EDA)
- Correlation analysis with target variable
- Feature categorization (continuous vs categorical)
- Visualization of top correlated features
- Relationship analysis through scatter plots and box plots

### Step 5: Data Wrangling & Preprocessing
- Feature engineering (creating new features)
- Outlier capping
- Handling rare categories
- Missing value imputation
- One-hot encoding for categorical variables

### Step 6: Model Building
- Train-test split (80-20)
- Linear Regression baseline model
- Model training and evaluation

### Step 7: Model Comparison & Feature Importance
- Cross-validation implementation
- Comparison of three models:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- Feature importance analysis

## 📊 Key Results

### Model Performance (5-fold Cross-Validation RMSE)
- **Linear Regression**: Baseline performance
- **Random Forest**: Improved accuracy with ensemble approach
- **XGBoost**: Best performing model with gradient boosting

### Top Influential Features
- Overall Quality of the house
- Above ground living area
- Total basement area
- Neighborhood location
- Garage area and capacity

## 🎨 Visualizations

The project includes comprehensive visualizations:
- Distribution plots for target variable
- Correlation heatmaps
- Scatter plots for continuous features
- Box plots for categorical features
- Feature importance bar charts

## 🚀 Usage

1. Ensure you have the `AmesHousing.csv` dataset in your project directory
2. Run the Jupyter notebook or Python script step by step
3. The model will output performance metrics and feature importance rankings

```python
# Example usage
import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('AmesHousing.csv')

# Follow the preprocessing steps
# Train your preferred model
# Make predictions
```

## 📋 Key Features

- **Comprehensive EDA**: In-depth analysis of housing features
- **Data Quality Handling**: Robust preprocessing pipeline
- **Multiple Model Comparison**: Evaluation of different algorithms
- **Feature Engineering**: Creation of meaningful derived features
- **Cross-Validation**: Reliable performance assessment
- **Feature Importance**: Insights into price-driving factors

## 🔍 Model Evaluation Metrics

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **Cross-Validation**: 5-fold validation for robust assessment

## 📝 Future Improvements

- Hyperparameter tuning for optimal model performance
- Advanced feature selection techniques
- Ensemble methods combining multiple models
- Implementation of neural networks
- Web application for real-time predictions

## 👨‍💻 Author

**Akanksha Waghamode**
- GitHub: https://github.com/akanksha3-3 
- LinkedIn: https://www.linkedin.com/in/akanksha-waghamode-25aa9724a/ 
- Email: akankshawaghamode2001@gmail.com

## 🙏 Acknowledgments

- Ames Housing Dataset creators
- Scikit-learn and XGBoost communities
- Open source Python ecosystem

---

⭐ **If you found this project helpful, please give it a star!** ⭐
