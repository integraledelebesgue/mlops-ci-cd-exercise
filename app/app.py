import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

app = FastAPI()

tokenizer = Tokenizer.from_file("model/tokenizer.json")
embedding_session = ort.InferenceSession("model/embedding.onnx")
classifier_session = ort.InferenceSession("model/classifier.onnx")


class Request(BaseModel):
    text: str


class Response(BaseModel):
    prediction: str


@app.post("/predict")
def predict(request: Request) -> Response:
    # tokenize input
    encoded = tokenizer.encode(request.text)

    # prepare numpy arrays for ONNX
    input_ids = np.array([encoded.ids])
    attention_mask = np.array([encoded.attention_mask])

    # run embedding inference
    embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    embeddings = embedding_session.run(None, embedding_inputs)[0]

    # run classifier inference
    classifier_input_name = classifier_session.get_inputs()[0].name
    classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
    prediction = classifier_session.run(None, classifier_inputs)[0]

    return Response(prediction="positive" if prediction == 1 else "negative")
