#!/usr/bin/env python3

import config
import pickle
import sys
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

sys.path.insert(0, config.ROOT)

from pipeline import SearchEngine

with open(config.MATRIX_EMBEDDING_PATH, "rb") as f : 
    matrix = pickle.load(f)

with open(config.LABELS_PATH, "rb") as f : 
    labels = pickle.load(f)


# Instanciate Sentence transformer
model = SentenceTransformer(config.MODEL_FOLDER)


search_engine = SearchEngine(matrix, labels)

# Load model
app = FastAPI(title = "20 News groups one shot clasification")

@app.get("/")
async def index() : 
    return {"Hello" : "World"}

@app.post("/predict/<sentence>")
async def predict(sentence : str) : 
    try : 
        return search_engine.predict(
            sentence, 
            model = model
        )
    except Exception as exception :
         HTTPException(status_code=404, detail = exception)

