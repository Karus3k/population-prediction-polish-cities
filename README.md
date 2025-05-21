# 🧠 Population Prediction Model Based on Urban Data

Predicting city population based on urban features using Python and machine learning.

## 🎯 Project Goal

The aim of this project is to create a machine learning model that predicts the population of Polish cities based on features such as:

- Area of the city (in km²)
- Province (voivodeship)
- Population density

This project demonstrates a full data analysis pipeline: from data exploration and preprocessing to model training and evaluation.

---

## 🗃 Dataset

The dataset contains information about 40 cities in Poland and includes the following columns:

| Column Name         | Description                       |
|---------------------|-----------------------------------|
| `miasto`            | Name of the city                  |
| `powierzchnia_km2`  | Area in square kilometers         |
| `ludnosc`           | Population                        |
| `wojewodztwo`       | Province (voivodeship)            |

Additional features derived:

- `wojewodztwo_encoded`: Province encoded with `LabelEncoder`
- `gestosc_zaludnienia`: Population density (`ludnosc` / `powierzchnia_km2`)

---

## 🧰 Technologies Used

- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🔍 Key Analytical Steps

### 1. Data Preparation
- Removed duplicates
- Filled missing values
- Created new features

### 2. Exploratory Data Analysis (EDA)
- Population histogram
- Area vs population scatter plot
- Correlation matrix heatmap
- Province ranking by average population

### 3. Machine Learning Models
- **Linear Regression**
- **Random Forest Regressor**

**Evaluation metric**: Mean Absolute Error (MAE)

| Model               | MAE (approx.) |
|--------------------|---------------|
| Linear Regression  | 64,106        |
| Random Forest      | 41,597 ✅     |

---

## 📈 Key Visualizations

<details>
<summary>📊 Show example plots</summary>

![Population Histogram](images/population_histogram.png)
![Area vs Population](images/area_vs_population.png)
![Correlation Matrix](images/correlation_matrix.png)

</details>

---

## 🚀 Possible Extensions

- Add more cities and update dataset dynamically
- Export predictions to a CSV file
- Build a dashboard (e.g., Streamlit or Tableau)
- Deploy as a REST API (FastAPI or Flask)

---

## 👨‍💻 Author

**Grzegorz Karabela**  
_This project is part of my journey to become a professional Data Analyst._

📫 Feel free to connect with me on [LinkedIn](https://www.linkedin.com/) or explore more projects on [my GitHub](https://github.com/TWOJ_LOGIN).

---

## 📂 Project Structure

```bash
.
├── app.py                  # Main script with model code
├── dane_testowe.csv        # Input dataset
├── Notatnik.ipynb          # Jupyter Notebook (EDA + modeling)
├── README.md               # Project documentation
└── images/                 # Folder with visualization exports (optional)
