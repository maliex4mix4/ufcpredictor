#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string


# In[2]:


letters = list(string.ascii_lowercase)


# In[88]:


fi = pd.read_json('Data/fighter/fighters-a.json')
for letter in letters:
    if letter != 'a':
        data = pd.read_json(f'Data/fighter/fighters-{letter}.json')
        fi = fi.append(data, ignore_index=True, )


# In[89]:


fighters = pd.read_csv('Data/ufc_fighters.csv')
matches = pd.read_csv('Data/ufc_bouts.csv', index_col='bout_id')


# In[90]:


fi.sample(3)


# In[91]:


fi.height.nunique()


# In[92]:


fighters.head(2)


# In[31]:


matches.head(2)


# In[32]:


matches.isna().sum()


# In[33]:


# matches.info()


# ## Cleaning the matches data

# In[72]:


matches = matches.loc[matches["result"] == "win"].copy()


# In[73]:


matches['location'] = matches['location'].map(lambda x: x.split(',')[-1])


# In[74]:


matches['figth_year'] = matches['date'].map(lambda x: x.split('-')[0])


# In[75]:


matches['method'] = matches['method'].map(lambda x: x.split('-')[0])


# In[76]:


matches['attended'] = matches['attendance ']


# In[77]:


matches['fight_coutry'] = matches['location']


# In[78]:


matches.drop(["event_name", "date", "location", "attendance ", "result", "end_time"],
          axis=1, inplace=True)


# In[79]:


matches.head(2)


# In[80]:


# winner is always fighter 1 
matches["fighter1"].equals(matches["winner"])


# In[81]:


matches[matches["fighter1"] != matches['winner']]


# In[82]:


# randomly swap fighter1 and fighter2 for half of the dataset to create "negative cases"
swap_indices = np.random.choice(len(matches), size = len(matches) // 2, replace = False)
matches.iloc[swap_indices, [0, 1]] = matches.iloc[swap_indices, [1, 0]].values


# In[83]:


matches["winner"] = matches["winner"] == matches["fighter1"]
matches["winner"] = matches["winner"].astype(int)
matches["winner"].value_counts()


# In[84]:


matches


# In[85]:


matches[matches["fighter1"]=="Dong Hyun Kim"]


# In[86]:


# again, handel duplicate names
matches_clean = matches.copy()
for col in ["fighter1", "fighter2"]:
    matches_clean.loc[(matches_clean[col] == "Michael McDonald") &
                    (matches_clean["weight_class"] == "Light Heavyweight"), col] = "Michael McDonald 205"
    
    matches_clean.loc[(matches_clean[col] == "Dong Hyun Kim") &
                    (matches_clean["weight_class"] == "Lightweight"), col]= "Dong Hyun Kim 155"
    
    matches_clean.loc[(matches_clean[col] == "Tony Johnson") &
                    (matches_clean["weight_class"] == "Middleweight"), col] = "Tony Johnson 185"
    
    matches_clean.loc[(matches_clean[col] == "Mike Davis") &
                    (matches_clean["weight_class"] == "Featherweight"), col] = "Mike Davis 145"


# In[87]:


matches_clean[matches_clean["fighter1"] == "Dong Hyun Kim 155"]


# ## Cleaning fighters data

# In[94]:


# check if there are fighters with the same name
fighters[fighters.duplicated(subset="name", keep=False)]


# In[95]:


fighters.iloc[513, 1] = "Michael McDonald 205"
fighters.iloc[845, 1] = "Dong Hyun Kim 155"
fighters.iloc[954, 1] = "Tony Johnson 185"
fighters.iloc[1674, 1] = "Mike Davis 145"


# In[96]:


# use fighter names as index
fighters.drop("fighter_id", axis=1, inplace=True)
fighters.set_index("name", inplace=True)


# In[97]:


# Some fighters do not have statistics available, and we will remove those fighters.
fighers_clean = fighters.loc[~((fighters["SLpM"] == 0) &
                               (fighters["Str_Acc"] == "0%") & 
                               (fighters["SApM"] == 0) &
                               (fighters["Str_Def"] == "0%") &
                               (fighters["TD_Avg"] == 0) &
                               (fighters["TD_Acc"] == "0%") &
                               (fighters["TD_Def"] == "0%") &
                               (fighters["Sub_Avg"] == 0))].copy()


# In[98]:


print("{0} fighers in total, after clean up: {1} fighers".format(len(fighters), len(fighers_clean)))


# In[99]:


# add winning percentages
fighers_clean["win%"] = 100 * (fighers_clean["win"] / (fighers_clean["win"] +
                                                 fighers_clean["lose"] +
                                                 fighers_clean["draw"] +
                                                 fighers_clean["nc"]))
# change datatypes
percentages = ["Str_Acc", "Str_Def", "TD_Acc", "TD_Def"]
statistics = ["SLpM", "Str_Acc", "SApM", "Str_Def", "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg", "win%"]

fighers_clean.loc[:, percentages] = fighers_clean.loc[:, percentages].applymap(
    lambda x: x.replace("%", ""))

fighers_clean.loc[:, statistics] = fighers_clean.loc[:, statistics].astype(np.float32)


# In[100]:


fighers_clean.head()


# In[101]:


# use born year only
def get_year(dob):
    if pd.isna(dob):
        return "unk"
    return dob.split(", ")[-1]

fighers_clean["born_year"] = fighers_clean["dob"].map(lambda dob: get_year(dob))


# In[102]:


fighers_clean.head()


# In[103]:


# remove unused columns
fighers_clean.drop(["weight", "dob", "last_updated"], inplace=True, axis=1)


# In[104]:


fighers_clean.head()


# ## Combining both data

# In[106]:


all_fighter_names = fighers_clean.index.values.tolist()

matches_clean = matches_clean.loc[(matches_clean["fighter1"].isin(all_fighter_names)) &
                              (matches_clean["fighter2"].isin(all_fighter_names))]


# In[107]:


matches_clean.reset_index(inplace=True, drop=True)


# In[109]:


matches_clean.head()


# In[110]:


fighter1_data = fighers_clean.loc[matches_clean["fighter1"]]
fighter1_data = fighter1_data.add_suffix('_fighter1')
fighter2_data = fighers_clean.loc[matches_clean["fighter2"]]
fighter2_data = fighter2_data.add_suffix('_fighter2')


# In[111]:


fighter1_data.head()


# In[112]:


fighter2_data.head()


# In[113]:


len(fighter1_data), len(fighter2_data)


# In[114]:


fighter1_data.reset_index(inplace=True, drop=True)
fighter2_data.reset_index(inplace=True, drop=True)
combined = pd.concat([matches_clean, fighter1_data, fighter2_data], axis=1, sort=False)


# In[115]:


combined.head()


# In[116]:


len(combined)


# In[117]:


combined.to_csv("Data/prediction_data.csv")


# In[118]:


fi.to_csv("Data/fighters_scraped.csv")
# fighters_clean.to_csv("Data/fighters.csv")


# In[ ]:




