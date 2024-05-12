# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:00:47 2024

@author: khaws
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):

    state:str
    district:str
    market:str
    commodity:str
    variety:str
    month:float 
    year:int
    day:int
    


model = pickle.load(open('price_prediction_trained2.sav', 'rb'))
preprocessor = pickle.load(open('processor2.sav', 'rb'))

@app.post('/predict')
def pred(input_parameters: model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    state = input_dictionary['state']
    district = input_dictionary['district']
    market = input_dictionary['market']
    commodity = input_dictionary['commodity']
    variety = input_dictionary['variety']
    month = input_dictionary['month']
    year = input_dictionary['year']
    day = input_dictionary['day']

    state = preprocessor['state'].transform([state])[0]
    district = preprocessor['district'].transform([district])[0]
    market = preprocessor['market'].transform([market])[0]
    commodity = preprocessor['commodity'].transform([commodity])[0]
    variety = preprocessor['variety'].transform([variety])[0]
    month = preprocessor['month'].transform([month])[0]
    


    prediction = model.predict([[state, district, market, commodity, variety, month, year, day]])
    
    return prediction[0]
