import numpy as np
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Step 1: Connect to Google Sheets
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    "apidatascience-455502-8907fd6a3a23.json", scope
)
client = gspread.authorize(creds)

# Open Google Sheet
spreadsheet = client.open_by_url(
    "https://docs.google.com/spreadsheets/d/1Lxes2shJYq1YqrMMlyAHkxfZsRRyM8eBZg6bI9p7I9s/edit?gid=0"
)
sheet = spreadsheet.sheet1
data = sheet.get_all_values()

# Convert to DataFrame and Fix Column Names
df = pd.DataFrame(data)
df.columns = df.iloc[0]  # Use first row as headers
df = df[1:]  # Remove first row from data
df = df.rename(columns=lambda x: x.strip())  # Remove spaces in column names
df.reset_index(drop=True, inplace=True)

# Print column names to debug
print("Columns in DataFrame:", df.columns)

# Convert Data Types
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df["Temperature(*C)"] = pd.to_numeric(df["Temperature(*C)"], errors="coerce")
df["Humidity (%)"] = pd.to_numeric(df["Humidity (%)"], errors="coerce")

# Drop invalid rows
df.dropna(inplace=True)

# Convert timestamps to numerical days
df["Day"] = (df["Timestamp"] - df["Timestamp"].min()).dt.days

# Train Linear Regression Models
X = df[["Day"]]
y_temp = df["Temperature(*C)"]
y_hum = df["Humidity (%)"]

X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
    X, y_temp, test_size=0.2, random_state=42
)
X_train_hum, X_test_hum, y_train_hum, y_test_hum = train_test_split(
    X, y_hum, test_size=0.2, random_state=42
)

model_temp = LinearRegression().fit(X_train_temp, y_train_temp)
model_hum = LinearRegression().fit(X_train_hum, y_train_hum)

# Predict Future Values
last_day = df["Day"].max()
future_days = np.array([last_day + i for i in range(1, 4)]).reshape(-1, 1)

future_temp = model_temp.predict(future_days)
future_hum = model_hum.predict(future_days)

# Convert Days Back to Dates
future_dates = [
    df["Timestamp"].min() + timedelta(days=int(day)) for day in future_days.flatten()
]

# Display Predictions
print("\nTemperature Predictions:")
for date, temp in zip(future_dates, future_temp):
    print(f"{date.strftime('%Y-%m-%d')}: {temp:.2f}°C")

print("\nHumidity Predictions:")
for date, hum in zip(future_dates, future_hum):
    print(f"{date.strftime('%Y-%m-%d')}: {hum:.2f}%")

# Plot Predictions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df["Day"], df["Temperature(*C)"], label="Actual", color="blue")
plt.plot(future_days, future_temp, "r-", label="Predicted")
plt.title("Temperature Prediction")
plt.xlabel("Days since first record")
plt.ylabel("Temperature (°C)")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(df["Day"], df["Humidity (%)"], label="Actual", color="green")
plt.plot(future_days, future_hum, "r-", label="Predicted")
plt.title("Humidity Prediction")
plt.xlabel("Days since first record")
plt.ylabel("Humidity (%)")
plt.legend()

plt.tight_layout()
plt.show()
