import json
import string
import pandas as pd
import json

df1 = pd.read_json('Data/ufc_fighters.json')

df = df1.drop_duplicates(subset="name", keep="first", inplace=False)

saving = df.fillna(0.00, inplace=False)

# print(saving.isna().sum())

saving.to_csv('Data/fighters.csv')

print('saved')