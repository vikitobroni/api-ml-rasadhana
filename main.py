import os
import io
import cv2
import json
import httpx
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
from google.cloud import storage
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Mengambil variabel dari .env
MONGO_URI = os.getenv("MONGO_URI")
GCLOUD_BUCKET_NAME = os.getenv("GCLOUD_BUCKET_NAME")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Fungsi untuk memuat model dari Google Cloud Storage
def load_model_from_gcs(bucket_name, model_path):
    """Load model dari Google Cloud Storage"""
    client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)  # Menggunakan kredensial GCS
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    model_file = io.BytesIO()
    try:
        blob.download_to_file(model_file)
        model_file.seek(0)
        model = tf.keras.models.load_model(model_file)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Fungsi untuk memuat dataset resep dari Cloud Storage
def load_recipe_from_gcs(bucket_name, recipe_path):
    """Load recipe dataset (JSON) from Google Cloud Storage"""
    client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)  # Menggunakan kredensial GCS
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(recipe_path)
    recipe_file = io.BytesIO()
    try:
        blob.download_to_file(recipe_file)
        recipe_file.seek(0)
        recipe_data = json.load(recipe_file)  # Correct JSON loading
        return recipe_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading recipe dataset: {str(e)}")

# Memuat model dari Cloud Storage (Ganti dengan nama bucket dan path model yang benar)
model = load_model_from_gcs(GCLOUD_BUCKET_NAME, 'model/model_food_classification2.h5')

# Memuat dataset resep dari Cloud Storage
recipe_dataset = load_recipe_from_gcs(GCLOUD_BUCKET_NAME, 'dataset/cleaned_dataset.json')

# Nama kelas (sesuaikan dengan kelas yang ada dalam model Anda)
class_names = ['caberawit', 'tomat', 'wortel', 'tempe', 'bawangputih', 'dagingsapi', 'kentang', 'dagingayam', 'bawang merah', 'telurayam']

# Fungsi untuk memuat gambar dari URL secara asynchronous
async def load_image_from_url(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 200:
            img_data = response.content
            img = Image.open(io.BytesIO(img_data))
            img = img.convert("RGB")  # Pastikan gambar dalam format RGB
            return np.array(img)
        else:
            raise HTTPException(status_code=404, detail="Gambar tidak ditemukan di URL")

# Fungsi untuk mengubah gambar menjadi array numpy dan melakukan preprocessing
def predict_single_image(model, img_data, target_size=(224, 224)):
    # Mengubah gambar menjadi array dan melakukan preprocessing
    img_resized = cv2.resize(img_data, target_size)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch

    # Prediksi dengan model
    predictions = model.predict(img_array)

    # Ambil kelas dengan confidence tertinggi
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence

# Fungsi untuk mendapatkan resep berdasarkan bahan yang diprediksi
def get_recipe_by_ingredient(ingredient):
    return recipe_dataset.get(ingredient, "Resep tidak ditemukan.")

# Inisialisasi aplikasi FastAPI
app = FastAPI()

@app.get("/")
def index():
    return "Hello world from ML endpoint!"

@app.post("/predict_latest_image/{user_id}")
async def predict_latest_image(user_id: str):
    try:
        # Ambil foto terbaru berdasarkan userId
        latest_photo_data = await get_latest_photo(user_id)  # Mendapatkan data foto terbaru dari MongoDB
        photo_url = latest_photo_data['photoUrl']  # URL foto terbaru yang disimpan di MongoDB

        # Baca gambar langsung dari URL
        photo_data = await load_image_from_url(photo_url)

        # Prediksi dengan model
        predicted_class_id, confidence = predict_single_image(model, photo_data, target_size=(224, 224))
        class_name = class_names[predicted_class_id]
        
        # Dapatkan resep berdasarkan kelas yang diprediksi
        recipe = get_recipe_by_ingredient(class_name)

        # Kembalikan hasil prediksi dan resep
        return JSONResponse(content={"class": class_name, "confidence": confidence * 100, "recipe": recipe})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Fungsi untuk mendapatkan foto terbaru berdasarkan userId dari MongoDB
async def get_latest_photo(user_id: str):
    try:
        # Menggunakan httpx untuk memanggil API yang ada di backend Node.js (Express)
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://be-rasadhana-245949327575.asia-southeast2.run.app/photo/latest/{user_id}")  # Gantilah URL dengan API yang sesuai
            if response.status_code == 200:
                return response.json()  # Mengembalikan data foto yang ditemukan
            else:
                raise HTTPException(status_code=404, detail="Foto tidak ditemukan")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    port = os.getenv("PORT", 8080)  # Menggunakan PORT dari .env
    uvicorn.run(app, host='0.0.0.0', port=port)
