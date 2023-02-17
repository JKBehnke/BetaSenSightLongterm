import numpy as np
import pandas as pd
import pickle
import json


tfr = np.ones(100_000)
# print(f"{tfr = }")

filepath = "tfr.pickle"
with open(filepath, "wb") as file:
    pickle.dump(tfr, file)

with open(filepath, "rb") as file:
    new_tfr = pickle.load(file)

# print(f"{new_tfr = }")

df = pd.DataFrame(
    tfr,
    columns=["power"],
)
print(df.head())

filepath = "tfr.json.zip"
df.to_json(filepath)
