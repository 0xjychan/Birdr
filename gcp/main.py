from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = "birdr"
class_names = ["Not a Bird", "Bird"]

model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def class_label(x):
  if x>=0.5:
    return 1
  else:
    return 0

def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/birdr_model0.h5",
            "/tmp/birdr_model0.h5"
        )
        model = tf.keras.models.load_model("/tmp/birdr_model0.h5")
    
    image = request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((224,224)))
    image_array = tf.expand_dims(image,0)
    prediction = model.predict(image_array).flatten()

    predicted_class = class_names[class_label(prediction[0])]
    confidence = round(100*prediction[0],2)

    return {'class':predicted_class,'confidence':float(confidence)}



    





