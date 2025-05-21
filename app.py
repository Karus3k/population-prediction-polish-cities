# === PROJECT: Population Prediction Model Based on Urban Data ===

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# === Load the data ===
dane = pd.read_csv("dane_testowe.csv")

# === Clean the data ===
dane = dane.drop_duplicates()
dane = dane.fillna(0)

# === Encode provinces ===
le = LabelEncoder()
dane['wojewodztwo_encoded'] = le.fit_transform(dane['wojewodztwo'])

# === Add column: population density ===
dane['gestosc_zaludnienia'] = dane['ludnosc'] / dane['powierzchnia_km2']

# === Data exploration ===
plt.hist(dane['ludnosc'], bins=10, edgecolor='black')
plt.title('Distribution of City Population')
plt.xlabel('Population')
plt.ylabel('Number of Cities')
plt.tight_layout()
plt.show()

plt.scatter(dane['powierzchnia_km2'], dane['ludnosc'])
plt.title('City Area vs. Population')
plt.xlabel('Area (km²)')
plt.ylabel('Population')
plt.grid(True)
plt.tight_layout()
plt.show()

sns.heatmap(dane.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# === Province ranking by average population ===
ranking = dane.groupby('wojewodztwo')[
    'ludnosc'].mean().sort_values(ascending=False)
print("Province ranking by average population:")
print(ranking)

# === Prepare data for modeling ===
X = dane[['powierzchnia_km2', 'wojewodztwo_encoded', 'gestosc_zaludnienia']]
y = dane['ludnosc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# === Model 1: Linear Regression ===
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
print("\n📉 LinearRegression - MAE:", round(mae_linear))

# === Model 2: Random Forest ===
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print("🌲 RandomForest - MAE:", round(mae_rf))
