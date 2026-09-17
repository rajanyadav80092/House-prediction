import pandas as pd
import joblib
from pydantic import BaseModel,Field
from fastapi import FastAPI,HTTPException


app=FastAPI()

model=joblib.load("house_model.joblib")
features=joblib.load("house_features.joblib")

class Housefeatures(BaseModel):
    MedInc:float=Field(gt=0,description="Media Income of NeighbourHood")
    HouseAge:float=Field(gt=0,description="Average age of house")
    AvgRms:float=Field(gt=0,description="Average rooms")
    AvgBedrms:float=Field(gt=0,description="Average bedrooms")
    Populations:float=Field(gt=0,description="Total population")
    AveOccup:float=Field(gt=0,description="Average distance of rooms")
    Latitude:float=Field(ge=32,le=42,description="Latitude")
    Longitude:float=Field(ge=-125,le=-114,description="Longitude")
    

@app.get("/")
def home():
    return {
        "message":"california house prediction app",
        "status":"running",
        "endpoint":"Send post request to prediction"
    }


@app.get("/health")
def health():
    return {
        "status":"running",
        "model":"RandomForestRegressor",
        "features":"feature",
        "avg_error":"$39000"
    }

# prediction
@app.post("/predict")
def predict(house:Housefeatures):
    try:
        input_data=pd.DataFrame([{
            "MedInc":house.MedInc,
            "HouseAge":house.HouseAge,
            "AveRooms":house.AvgRms,
            "AveBedrms":house.AvgBedrms,
            "Population":house.Populations,
            "AveOccup":house.AveOccup,
            "Latitude":house.Latitude,
            "Longitude":house.Longitude
        }])
        
        predicted=model.predict(input_data)[0]
        price_usd=predicted*100000
        return {
            "predicted_price":f"{price_usd :,.0f}",
            "predicted_price_short":f"${predicted:,.2f} hundred thousand",
            "findence_range":f"$ {price_usd-39000:,.0f} to ${price_usd+39000 :,.0f}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed : {str(e)}"
        )
