# California House Price Prediction API

## Features
- House price prediction
- Pydantic validation
- CSV batch prediction
- CSV result download
- Health check endpoint
- Random Forest Regression
- FastAPI

## Endpoints

GET /
GET /health
POST /predict
POST /predict-file

# required data 
{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AvgRms": 6.9484,
  "AvgBedrms": 1.023,
  "Populations": 322,
  "AveOccup": 2.55,
  "Latitude": 37.88,
  "Longitude": -122.25
}