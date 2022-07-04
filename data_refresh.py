import json
import string
import pandas as pd
from ufcpy import Fighter

list_of_fighters = []
letters = list(string.ascii_lowercase)
real_list = []

for x in letters:
    fi = json.load(open(f"./Data/fighter/fighters-{x}.json", "r"))
    list_of_fighters.append(fi)

for x in list_of_fighters:
    for i in x:
        i['name'] = str(i['fname'])+" "+i['lname']
        real_list.append(i)

with open("results.json", 'w+') as f:
    f.write(json.dumps(real_list, indent=2))
    f.close()
