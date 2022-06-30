from copy import deepcopy
import pandas as pd
import pickle
import numpy as np

pickle_in = open("/Users/janikbischoff/Documents/Uni/MLog/Semester 1/FSIS/api/data/df_cat.pickle", "rb")
df_cat = pickle.load(pickle_in)
df_labeled = df_cat
df_cat_val = df_cat.copy(deepcopy)

for cat in df_cat:
  labels = []
  classes = df_cat[cat].unique()
  for value in df_cat[cat]:
    indexes = np.where(classes == value)
    index = indexes[0][0]
    labels.append(index)
  df_labeled[cat] = labels

df_labeled = df_labeled.reset_index()
df_cat_val = df_cat_val.reset_index()

name_mapping = {}
zip_customer_mapping = {}
customer_state_mapping = {}
zip_seller_mapping = {}
seller_state_mapping = {}
day_mapping = {}

for index, row in df_cat_val.iterrows():
    if row["product_category_name"] not in name_mapping:
        name_mapping[row["product_category_name"]] = df_labeled.iloc[index]["product_category_name"]
        
    elif row["customer_zip_code"] not in zip_customer_mapping:
        zip_customer_mapping[row["customer_zip_code"]] = df_labeled.iloc[index]["customer_zip_code"]

    elif row["customer_state"] not in customer_state_mapping:
        customer_state_mapping[row["customer_state"]] = df_labeled.iloc[index]["customer_state"]

    elif row["seller_zip_code"] not in zip_seller_mapping:
        zip_seller_mapping[row["seller_zip_code"]] = df_labeled.iloc[index]["seller_zip_code"]

    elif row["seller_state"] not in seller_state_mapping:
        seller_state_mapping[row["seller_state"]] = df_labeled.iloc[index]["seller_state"]

    elif row["order_weekday"] not in day_mapping:
        day_mapping[row["order_weekday"]] = df_labeled.iloc[index]["order_weekday"]

to_save = [name_mapping, zip_customer_mapping, customer_state_mapping, zip_seller_mapping, seller_state_mapping, day_mapping]
names = ["name_mapping", "zip_customer_mapping", "customer_state_mapping", "zip_seller_mapping", "seller_state_mapping", "day_mapping"]
to_save_zip = zip(names, to_save)

for name, dictionary in to_save_zip:
    with open(f'data/mappings/{name}.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


