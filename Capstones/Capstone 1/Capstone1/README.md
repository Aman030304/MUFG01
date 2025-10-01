# Manufacturing Equipment Output Prediction Project

## 🎯 Project Objective
Predict manufacturing equipment output (`Parts_Per_Hour`) using advanced linear regression models to optimize production efficiency and enable predictive maintenance.

## 📊 Dataset Description
- **Size**: 1000 manufacturing samples with 19 features
- **Target Variable**: `Parts_Per_Hour` (continuous numerical output)
- **Features**:
  - **Numerical**: Injection temperature, pressure, cycle time, cooling time, material viscosity, ambient temperature, machine age, operator experience, maintenance hours, efficiency metrics
  - **Categorical**: Shift (Day/Evening/Night), Machine Type (A/B/C), Material Grade (Economy/Standard/Premium), Day of Week
  - **Derived**: Temperature-pressure ratio, total cycle time, efficiency score, machine utilization
- **Data Quality**: Contains some missing values that are handled during preprocessing

## 🏗️ Project Architecture
```
Data Science prj/
├── data/                                    # Raw dataset
│   └── manufacturing_dataset_1000_samples.csv
├── models/                                  # Trained models
│   ├── linear_regression.pkl
│   ├── ridge_regression.pkl
│   ├── lasso_regression.pkl
│   ├── best_*.pkl                          # Best performing model
│   └── model_comparison.csv                # Model performance comparison
├── notebooks/                               # Analysis and experimentation
│   └── eda.ipynb                          # Comprehensive EDA notebook
├── src/                                    # Source code modules
│   ├── data_loader.py                     # Data loading utilities
│   ├── preprocessing.py                    # Data cleaning and scaling
│   ├── train_test_splitter.py             # Train-test data splitting
│   ├── model.py                           # Model training and comparison
│   └── evaluate.py                        # Model evaluation and visualization
├── main.py                                 # Main execution pipeline
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

## 🔄 Complete Workflow Pipeline

### 1. **Data Loading & Exploration**
- Load manufacturing dataset from `data/`
- Comprehensive EDA in Jupyter notebook
- Statistical analysis and missing value assessment

### 2. **Data Preprocessing**
- Handle missing values (impute with mean for numerical features)
- Drop timestamp column (not used for prediction)
- One-hot encode categorical variables
- **Feature Scaling**: Apply StandardScaler to numerical features
- Separate features (X) and target (y)

### 3. **Model Training**
- **Linear Regression**: Baseline model without regularization
- **Ridge Regression**: L2 regularization to prevent overfitting
- **Lasso Regression**: L1 regularization for feature selection
- Train all models on scaled training data
- Log top features by coefficient importance

### 4. **Model Evaluation & Comparison**
- Evaluate all models on test set
- Calculate MSE and R² metrics
- Generate comparison dataframe
- **Visualization**: Predicted vs Actual plots with diagonal reference line
- Save best performing model automatically

### 5. **Model Persistence**
- Save all trained models as `.pkl` files
- Export model comparison results to CSV
- Store best model with descriptive naming

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. **Clone/Download** the project to your local machine
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Execution
1. **Run the complete pipeline**:
   ```bash
   python main.py
   ```

2. **Explore the data** (optional):
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

## 📈 Expected Outputs

### Console Output
- Data loading and preprocessing progress
- Model training status and feature importance
- Model comparison table (MSE, R² scores)
- Best model identification and performance metrics

### Generated Files
- **Models**: `models/best_*.pkl` (best performing model)
- **Comparison**: `models/model_comparison.csv` (all model results)
- **Visualization**: Predicted vs Actual plot for best model

## 🛠️ Technologies & Libraries

- **Core ML**: scikit-learn (Linear, Ridge, Lasso regression)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib
- **Development**: Jupyter notebooks, logging

## 💼 Business Value

### **Predictive Maintenance**
- Anticipate equipment performance degradation
- Schedule maintenance before critical failures
- Reduce unplanned downtime

### **Production Planning**
- Forecast hourly production capacity
- Optimize manufacturing schedules
- Improve resource allocation

### **Quality Control**
- Identify factors affecting production efficiency
- Monitor equipment performance trends
- Ensure consistent output quality

### **Cost Optimization**
- Minimize production delays
- Reduce maintenance costs
- Improve overall equipment effectiveness (OEE)

## 🔍 Key Features

- **Automated Pipeline**: End-to-end execution with single command
- **Model Comparison**: Systematic evaluation of multiple algorithms
- **Feature Scaling**: Proper preprocessing for optimal model performance
- **Visualization**: Clear plots for model interpretation
- **Model Persistence**: Ready-to-deploy trained models
- **Comprehensive EDA**: Detailed data exploration and insights

## 📝 Notes

- The pipeline automatically handles missing values and categorical encoding
- Feature scaling is applied only to numerical features, preserving one-hot encoded categorical variables
- The best performing model is automatically identified and saved
- All models are evaluated on the same test set for fair comparison
- Logging provides detailed progress tracking throughout the pipeline

---

**Run `python main.py` to execute the complete manufacturing equipment output prediction pipeline!**
