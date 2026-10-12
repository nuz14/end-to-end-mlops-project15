
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("iris.csv")
X, y = df.iloc[:, :4], df["target"]

m = RandomForestClassifier()
m.fit(X, y)
joblib.dump(m, "model.joblib")
print("Model trained and saved")
