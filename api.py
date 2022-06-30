from flask import Flask, jsonify, request
import pickle
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import json

KEY = "bf698b456ad4cebada870914b2021fab"
def construct_api_call(query):
    url = f"http://api.positionstack.com/v1/forward?access_key={KEY}&query={query}"
    r = requests.get(url)
    body = r.json()
    try: 
        return [body["data"][0]["latitude"], body["data"][0]["longitude"]]
    except:
        return np.nan

def create_inputs(modelparams):
    columns_num = ["item_count", "price", "freight_value", "product_weight_g", "product_volume_cm^3", "delivery_distance", "order_hour_of_day"]
    df_num = pd.DataFrame(columns=columns_num)
    columns_cat = ["product_category_name", "customer_zip_code", "customer_state", "seller_zip_code", "seller_state", "order_weekday"]
    df_cat = pd.DataFrame(columns=columns_cat)
    
    order_weekday = modelparams["order_day"]
    names = modelparams["products"]
    customer_zip_code = modelparams["customer_zip"]
    customer_state = modelparams["customer_state"]
    customer_city = modelparams["customer_city"]
    seller_zip = modelparams["seller_zip"]
    seller_city = modelparams["seller_city"]
    seller_state = modelparams["seller_state"]

    names_labeled = [name_mapping[name] for name in names]
    customer_zip_code_labeled = zip_customer_mapping[customer_zip_code]
    customer_state_labeled = customer_state_mapping[customer_state]
    seller_zip_labeled = zip_seller_mapping[seller_zip]
    seller_state_labeled = seller_state_mapping[seller_state]

    for item in range(len(names)):
        row = {"product_category_name":names_labeled[item], "customer_zip_code":customer_zip_code_labeled, "customer_state":customer_state_labeled, "seller_zip_code":seller_zip_labeled, 
                "seller_state":seller_state_labeled, "order_weekday":order_weekday}
        add_row = pd.Series(row)
        
        df_cat = df_cat.append(add_row, ignore_index=True)
    df_labeled = df_cat
    print(df_labeled)
    c_0 = df_labeled[['product_category_name']].values
    c_0 = np.asarray(c_0).astype('float32')

    c_1 = df_labeled[['customer_zip_code']].values
    c_1 = np.asarray(c_1).astype('float32')

    c_2 = df_labeled[['customer_state']].values
    c_2 = np.asarray(c_2).astype('float32')

    c_3 = df_labeled[['seller_zip_code']].values
    c_3 = np.asarray(c_3).astype('float32')

    c_4 = df_labeled[['seller_state']].values
    c_4 = np.asarray(c_4).astype('float32')

    c_5 = df_labeled[['order_weekday']].values
    c_5 = np.asarray(c_5).astype('float32')

    item_count = 1
    prices = modelparams["prices"]
    weights = modelparams["weights"]
    volumes = modelparams["volumes"]
    order_hour_of_day = modelparams["order_hour"]
    query_seller = f"{seller_zip}, {seller_city}"
    query_customer = f"{customer_zip_code}, {customer_city}" 
    lat_lon_customer = construct_api_call(query_seller)
    lat_lon_seller = construct_api_call(query_customer)
    distance = get_manhatten_distance(lat_lon_customer, lat_lon_seller)

    for item in range(len(prices)):
        row = {"item_count":item_count, "price":prices[item], "freight_value":23, "product_weight_g":weights[item], 
                "product_volume_cm^3":volumes[item], "delivery_distance":distance, "order_hour_of_day":order_hour_of_day}
        add_row = pd.Series(row)
        
        df_num = df_num.append(add_row, ignore_index=True)
    ss = StandardScaler()
    x_standardized = ss.fit_transform(df_num)
    X = x_standardized
    X = np.asarray(X).astype('float32')

    return {"cat_input_0": c_0, "cat_input_1": c_1, "cat_input_2": c_2, "cat_input_3": c_3,"cat_input_4": c_4, "cat_input_5": c_5,"numeric_inputs": X}
     

def get_manhatten_distance(customer, seller):
    R = 6371000.785 

    phi_1 = customer[0] * (math.pi / 180)
    phi_2 = seller[0] * (math.pi / 180)
    delta_phi = (seller[0] - customer[0]) * (math.pi / 180)
    delta_lamda = (seller[1] - customer[1]) * (math.pi / 180)

    a = math.sin(delta_phi / 2) * math.sin(delta_phi / 2) + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lamda/2) * math.sin(delta_lamda/2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = R * c
    
    return d / 1000

with open('data/mappings/name_mapping.pickle', 'rb') as handle:
    name_mapping = pickle.load(handle)
with open('data/mappings/zip_customer_mapping.pickle', 'rb') as handle:
    zip_customer_mapping = pickle.load(handle)
with open('data/mappings/customer_state_mapping.pickle', 'rb') as handle:
    customer_state_mapping = pickle.load(handle)
with open('data/mappings/zip_seller_mapping.pickle', 'rb') as handle:
    zip_seller_mapping = pickle.load(handle)
with open('data/mappings/seller_state_mapping.pickle', 'rb') as handle:
    seller_state_mapping = pickle.load(handle)

app = Flask(__name__)


@app.route("/")
def index():
    return "<h1>Usage: url/predict/password/modelparams</h1>"

@app.route("/predict/<password>/<modelparams>")
def predict(password, modelparams):
    model = tf.keras.models.load_model("data/models/31days_all.h5")
    if password == "FSIS2022":
        if modelparams:
            print(json.loads(modelparams))
            inputs = create_inputs(json.loads(modelparams))
            pickle_in = open("data/y.pickle", "rb")
            df_y = pickle.load(pickle_in)
            y = df_y.to_numpy()
            enc = OneHotEncoder(handle_unknown='ignore')
            y = y.reshape(-1,1)
            enc.fit(y)
            
            predictions = model.predict(
                inputs
                )
            b = np.zeros_like(predictions)
            b[np.arange(len(predictions)), predictions.argmax(1)] = 1
            etas = enc.inverse_transform(b)
            return str(etas.ravel()[0])
        else:
            result = {"ETA":"10 Tage"}
        return jsonify(result)
    else:
        result = "Falscher API Key"
        return jsonify(result)


