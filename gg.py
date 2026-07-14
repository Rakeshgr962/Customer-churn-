import pandas as pd

df = pd.read_csv("data/telco.csv")

print(df.iloc[0].to_dict())