from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

MODEL0 = tf.keras.models.load_model('./birdr_model0.h5')
CLASS_NAME = ['Not a Bird', 'Bird']

app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_headers=['*'],
        allow_methods=['*'],
        allow_origins=['*'],
    )

@app.get('/')
async def home():
    return 'Server is running'

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    print(image)
    return image

def class_label(x):
  if x>=0.5:
    return 1
  else:
    return 0

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image = image[:,:,:3] # remove alpha channel if exist
    image = tf.image.resize(image, [224,224]) # resize to (224,224)
    image_batch = np.expand_dims(image,0)
    prediction = MODEL0.predict(image_batch).flatten()
    predicted_class = CLASS_NAME[class_label(prediction[0])]
    confidence = prediction[0]
    return {
        'class':predicted_class,
        'confidence': float(confidence)
    }



