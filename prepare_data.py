
import pandas as pd
from sklearn.datasets import load_iris

d = load_iris()
df = pd.DataFrame(d.data, columns=d.feature_names)
df["target"] = d.target
df.to_csv("iris.csv", index=False)
print("Dataset saved")
