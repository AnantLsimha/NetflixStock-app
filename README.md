# Netflix Stock Price Prediction App 📈

This project is a complete pipeline for **predicting Netflix's stock closing prices** using various regression models and providing a **Streamlit-based user interface** for real-time predictions.

---

## 📂 Dataset
- Source: `NFLX.csv` file (Netflix historical stock data)
- Columns used:
  - `Open`, `High`, `Low`, `Close`, `Volume`, `Date`
  - Engineered Features: `HL_diff`, `Price_range`, `Year`, `Month`, `Day`

## 🧪 Exploratory Data Analysis (EDA)
- Handled missing values
- Removed `Adj Close` if present
- Converted `Date` to datetime format
- Created new time-based features (`Year`, `Month`, `Day`)
- Feature Engineering:
  - `HL_diff = High - Low`
  - `Price_range = High - Open`
- Plotted:
  - Correlation heatmap
  - Time-series Close Price trend
  - Scatter plots of `Open`, `High`, `Low`, `Volume` vs `Close`

## 🧠 Models Used
All models trained on features excluding `Close` (target variable):

| Model               | Technique         | Scaling Applied |
|--------------------|-------------------|------------------|
| Linear Regression  | Simple linear     | ✅ Yes           |
| Ridge Regression   | L2 Regularization | ✅ Yes           |
| Lasso Regression   | L1 Regularization | ✅ Yes           |
| Random Forest      | Ensemble Tree     | ❌ No (not needed) |

### 🔢 Evaluation Metric:
- R² Score (coefficient of determination)

## 🖥️ Streamlit Interface
Interactive web app includes:
- Displays R² scores for all models
- Highlights best model
- Accepts user input for today’s stock features
- Predicts Close Price using selected model

### 🔍 Inputs for Prediction:
- `Open`, `High`, `Low`, `Volume`
- `Year`, `Month`, `Day`

### ⚙️ Prediction Logic:
Feature engineered inputs (`HL_diff`, `Price_range`) are computed and passed to the trained model.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Requirements (partial)
- pandas
- matplotlib
- seaborn
- scikit-learn
- streamlit

---

## 📸 Screenshots
_You can add screenshots here for better presentation._


---

## 📌 Notes
- Ensure `NFLX.csv` file is available at the specified path.
- Works best for educational and demonstration purposes.