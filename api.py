from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict_message

app = FastAPI()


class Message(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Spam Detection API is Running!"
    }


@app.post("/predict")
def predict(data: Message):

    result = predict_message(data.text)

    return {
        "message": data.text,
        "prediction": result
    }