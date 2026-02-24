# 🏠 California Housing Price Prediction — Linear Regression

A machine learning project that builds and evaluates a **Linear Regression** model using the California Housing dataset. This project demonstrates the complete ML workflow — from data exploration and preprocessing to model training, evaluation, and deployment through a simple prediction UI.

---

## 📌 Project Overview

This project focuses on predicting median house values based on demographic and geographic features. It covers the full machine learning lifecycle:

* Exploratory Data Analysis (EDA)
* Model Training using Linear Regression
* Evaluation using MAE, RMSE, and R²
* Model Serialization using Pickle
* Prediction Interface (CLI / Streamlit UI)

---

## 📖 Theory — Linear Regression

Linear Regression is a supervised machine learning algorithm used for predicting continuous numerical values. It works by finding the best-fitting straight line that represents the relationship between input features and the target variable.

In this project:

* The independent variables include features like **Median Income, House Age, Average Rooms, Population, Latitude, and Longitude**.
* The dependent variable is **MedHouseVal**, representing the median house value.

The model learns coefficients for each feature and predicts house prices using a linear equation. Linear Regression is commonly used as a baseline model because it is simple, interpretable, and computationally efficient.

---

## 🧠 Technologies Used

* Python
* pandas, numpy
* scikit-learn
* matplotlib, seaborn
* Streamlit
* VS Code

---

## 📊 Dataset

* **California Housing Dataset** (from scikit-learn)
* Target Variable: `MedHouseVal`
* Key Features:

  * MedInc
  * HouseAge
  * AveRooms
  * AveBedrms
  * Population
  * AveOccup
  * Latitude
  * Longitude

---

## ⚙️ Project Structure

```
california-housing-linear-regression/
│
├── task1_ml_linear_regression.ipynb
├── linear_regression_model.pkl
├── predict_ui.py
├── app.py
├── Report.pdf
└── requirements.txt
```

---

## 🚀 How to Run

### 1️⃣ Clone Repository

```
git clone https://github.com/Venkatvds/california-housing-linear-regression.git
cd california-housing-linear-regression
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run Prediction UI (Streamlit)

```
streamlit run app.py
```

---

## 📈 Model Evaluation

Performance is measured using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

See **Report.pdf** for detailed analysis and insights.

---

## 💡 Future Improvements

* Ridge / Lasso Regression
* Feature Scaling
* Hyperparameter Tuning
* Model Deployment Enhancements

---

## 👤 Author

**Venkata Durga Sai D**

---

⭐ If you find this project useful, feel free to star the repository!
