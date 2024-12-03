import os
import uvicorn
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import storage
import numpy as np
import io
import json
from PIL import Image

# Fungsi untuk memuat model dari Google Cloud Storage
def load_model_from_gcs(bucket_name, model_path):
    """Load model dari Google Cloud Storage"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    model_file = io.BytesIO()
    blob.download_to_file(model_file)
    model_file.seek(0)
    model = tf.keras.models.load_model(model_file)
    return model

# Fungsi untuk memuat dataset resep dari Cloud Storage
def load_recipe_from_gcs(bucket_name, recipe_path):
    """Load recipe dataset (JSON) from Google Cloud Storage"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(recipe_path)
    recipe_file = io.BytesIO()
    blob.download_to_file(recipe_file)
    recipe_file.seek(0)
    recipe_data = json.load(recipe_file)
    return recipe_data

# Memuat model dari Cloud Storage (Ganti dengan nama bucket dan path model yang benar)
model = load_model_from_gcs('rasadhana-app-ml', 'model/model_food_classification2.h5')

# Memuat dataset resep dari Cloud Storage
recipe_dataset = load_recipe_from_gcs('rasadhana-app-ml', 'dataset/cleaned_dataset.json')

# Nama kelas (sesuaikan dengan kelas yang ada dalam model Anda)
class_names = ['caberawit', 'tomat', 'wortel', 'tempe', 'bawangputih', 'dagingsapi', 'kentang', 'dagingayam', 'bawang merah', 'telurayam']

# Fungsi untuk mengubah gambar menjadi array numpy
def load_image_into_numpy_array(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.convert("RGB")  # Pastikan gambar dalam format RGB
    return np.array(image)

# Fungsi untuk mendapatkan resep berdasarkan bahan yang diprediksi
def get_recipe_by_ingredient(ingredient):
    return recipe_dataset.get(ingredient, "Resep tidak ditemukan.")

# Inisialisasi aplikasi FastAPI
app = FastAPI()

@app.get("/")
def index():
    return "Hello world from ML endpoint!"

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="File is not an image")

    try:
        # Baca dan proses gambar
        image_data = await file.read()
        image = load_image_into_numpy_array(image_data)
        image = tf.image.resize(image, (224, 224))  # Resize gambar sesuai ukuran model
        image = image / 255.0  # Normalisasi gambar
        image = tf.expand_dims(image, 0)  # Tambah dimensi batch

        # Prediksi dengan model
        predictions = model.predict(image)
        predicted_class_id = tf.argmax(predictions, axis=1).numpy()[0]
        class_name = class_names[predicted_class_id]
        
        # Dapatkan resep berdasarkan kelas yang diprediksi
        recipe = get_recipe_by_ingredient(class_name)

        return JSONResponse(content={"class": class_name, "recipe": recipe})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    port = os.environ.get("PORT", 8080)
    uvicorn.run(app, host='0.0.0.0', port=port)
