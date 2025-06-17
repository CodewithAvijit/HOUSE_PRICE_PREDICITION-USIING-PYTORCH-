from fastapi import FastAPI,Form
import torch 
from torch import nn
from sklearn.preprocessing import StandardScaler
import joblib
from fastapi.responses import PlainTextResponse
from typing import Literal
app = FastAPI()
mainroad_en=joblib.load("encoder/mainroad_std.pkl")
guestroom_en=joblib.load("encoder/guestroom_std.pkl")
aircon_en=joblib.load("encoder/airconditioning_std.pkl")
prefarea_en=joblib.load("encoder/prefarea_std.pkl")
furnished_en=joblib.load("encoder/furninsh_std.pkl")
scalex=joblib.load("scale_x.pkl")
scaley=joblib.load("scale_y.pkl")
model=nn.Linear(in_features=10,out_features=1)
model.load_state_dict(torch.load("model.pth"))
model.eval()
@app.get("/")
def read_root():
    return {"message": "This is a house price prediction model using PyTorch."}

@app.post("/predict")
def predict(
    mainroad: Literal['yes', 'no'] = Form(..., description="Need main road?"),
    guestroom: Literal['yes', 'no'] = Form(..., description="Need guest room?"),
    airconditioning: Literal['yes', 'no'] = Form(..., description="Need air conditioning?"),
    prefarea: Literal['yes', 'no'] = Form(..., description="Preferred area?"),
    furnished: Literal['furnished', 'semi-furnished', 'unfurnished'] = Form(..., description="Furnishing status?"),
    area: int = Form(..., description="Total area in sq.ft"),
    bedrooms: int = Form(..., description="Number of bedrooms"),
    bathrooms: int = Form(..., description="Number of bathrooms"),
    stories: int = Form(..., description="Number of stories"),
    parking: int = Form(..., description="Parking capacity")
):
    mainroad=(mainroad_en.transform([[mainroad]])[0][0])
    guestroom=(guestroom_en.transform([[guestroom]])[0][0])
    airconditioning=(aircon_en.transform([[airconditioning]])[0][0])
    prefarea=(prefarea_en.transform([[prefarea]])[0][0])
    furnished=(furnished_en.transform([[furnished]])[0][0])
    input=scalex.transform([[mainroad,guestroom,airconditioning,prefarea,furnished,area,bedrooms,bathrooms,stories,parking]])
    from fastapi import FastAPI,Form
from sklearn.preprocessing import StandardScaler
import joblib
from fastapi.responses import PlainTextResponse
from typing import Literal
app = FastAPI()
mainroad_en=joblib.load("encoder/mainroad_std.pkl")
guestroom_en=joblib.load("encoder/guestroom_std.pkl")
aircon_en=joblib.load("encoder/airconditioning_std.pkl")
prefarea_en=joblib.load("encoder/prefarea_std.pkl")
furnished_en=joblib.load("encoder/furninsh_std.pkl")
scalex=joblib.load("scale_x.pkl")
scaley=joblib.load("scale_y.pkl")
@app.get("/")
def read_root():
    return {"message": "This is a house price prediction model using PyTorch."}

@app.post("/predict")
def predict(
    mainroad: Literal['yes', 'no'] = Form(..., description="Need main road?"),
    guestroom: Literal['yes', 'no'] = Form(..., description="Need guest room?"),
    airconditioning: Literal['yes', 'no'] = Form(..., description="Need air conditioning?"),
    prefarea: Literal['yes', 'no'] = Form(..., description="Preferred area?"),
    furnished: Literal['furnished', 'semi-furnished', 'unfurnished'] = Form(..., description="Furnishing status?"),
    area: int = Form(..., description="Total area in sq.ft"),
    bedrooms: int = Form(..., description="Number of bedrooms"),
    bathrooms: int = Form(..., description="Number of bathrooms"),
    stories: int = Form(..., description="Number of stories"),
    parking: int = Form(..., description="Parking capacity")
):
    mainroad=float(mainroad_en.transform([[mainroad]]))
    guestroom=float(guestroom_en.transform([[guestroom]]))
    airconditioning=float(aircon_en.transform([[airconditioning]]))
    prefarea=float(prefarea_en.transform([[prefarea]]))
    furnished=float(furnished_en.transform([[furnished]]))
    input=scalex.transform([[mainroad,guestroom,airconditioning,prefarea,furnished,area,bedrooms,bathrooms,stories,parking]])
    X_scaled = scalex.transform([[
        mainroad, guestroom, airconditioning, prefarea, furnished,
        area, bedrooms, bathrooms, stories, parking
    ]])
    value=torch.tensor(X_scaled,dtype=torch.float)
    with torch.no_grad():
     value=model(value).detach().numpy()
    price = float(scaley.inverse_transform(value)[0, 0])

    return {
        "PRICE":price
    }