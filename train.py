from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split

print("loading dataset")

data=fetch_california_housing()

x=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=RandomForestRegressor(
    random_state=42,
    n_estimators=100
)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Average error : ${mae*100000:,.0f}")

joblib.dump(model,"house_model.joblib")
joblib.dump(list(x.columns),"house_features.joblib")