import os
import io
import cv2
import json
import httpx
import uvicorn
import tempfile
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
from google.cloud import storage

# Nonaktifkan opsi oneDNN (jika diperlukan)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Mendapatkan variabel lingkungan dari .env
GCLOUD_BUCKET_NAME_ML = os.getenv("GCLOUD_BUCKET_NAME_ML")

# uncomen jika ingin menggunakan service account
# GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Nama kelas (sesuaikan dengan model Anda)
class_names = ['caberawit', 'tomat', 'wortel', 'tempe', 'bawangputih', 'dagingsapi', 'kentang', 'dagingayam', 'bawang merah', 'telurayam']


# Inisialisasi client GCS tanpa file kredensial
client = storage.Client()

# Fungsi untuk memuat model dari GCS
def load_model_from_gcs(bucket_name, model_path):
    try:
        # client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)  # Load credentials jika ingin menggunakan service account
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(model_path)
        model_file = io.BytesIO()
        blob.download_to_file(model_file)
        model_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file.write(model_file.read())
            tmp_file_path = tmp_file.name
        return tf.keras.models.load_model(tmp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model from GCS: {str(e)}")

# Fungsi untuk memuat dataset resep dari Google Cloud Storage
def load_recipe_from_gcs(bucket_name, recipe_path):
    """Load dataset resep dari Google Cloud Storage"""
    try:
        # client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)  # Load credentials jika ingin menggunakan service account
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(recipe_path)
        
        # Download dataset resep dan simpan ke memory
        recipe_file = io.BytesIO()
        blob.download_to_file(recipe_file)
        recipe_file.seek(0)
        
        # Load dataset resep dari file dalam memory
        recipe_data = json.load(recipe_file)
        return recipe_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading recipe dataset from GCS: {str(e)}")

# Ganti dengan nama bucket dan path file yang sesuai di Google Cloud Storage
bucket_name = GCLOUD_BUCKET_NAME_ML
model_path = "model/model_food_classification2.h5"
recipe_path = "dataset/cleaned_dataset.json"

# Memuat model dan dataset dari Google Cloud Storage
model = load_model_from_gcs(bucket_name, model_path)
recipe_dataset = load_recipe_from_gcs(bucket_name, recipe_path)

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
    img_resized = cv2.resize(img_data, target_size)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence

# Fungsi untuk mendapatkan resep berdasarkan bahan yang diprediksi
def get_recipe_by_ingredient(ingredient):
    """Mencari resep berdasarkan bahan yang diprediksi"""
    for recipe in recipe_dataset:
        # Cek apakah ingredient yang diprediksi ada dalam list Ingredients
        if ingredient.lower() in recipe["Ingredients"].lower():
            return {
                "Title": recipe["Title"],
                "Ingredients": recipe["Ingredients"],
                "Steps": recipe["Steps"],
                "URL": recipe["URL"]
            }
    return {"message": "Resep tidak ditemukan."}

# Inisialisasi aplikasi FastAPI
app = FastAPI()

@app.get("/")
def index():
    return "Hello world from ML endpoint!"

@app.post("/predict_latest_image/{user_id}")
async def predict_latest_image(user_id: str):
    try:
        # Bersihkan `user_id` dari karakter whitespace atau newline
        user_id = user_id.strip()
        
        # Ambil foto terbaru berdasarkan userId
        latest_photo_data = await get_latest_photo(user_id)
        photo_url = latest_photo_data['photoUrl']  # Pastikan key 'photoUrl' ada

        # Baca gambar langsung dari URL
        photo_data = await load_image_from_url(photo_url)

        # Prediksi dengan model
        predicted_class_id, confidence = predict_single_image(model, photo_data, target_size=(224, 224))
        class_name = class_names[predicted_class_id]

        # Konversi confidence dari float32 ke float
        confidence = float(confidence)

        # Dapatkan resep berdasarkan kelas yang diprediksi
        recipe = get_recipe_by_ingredient(class_name)

        return JSONResponse(content={"class": class_name, "confidence": confidence * 100, "recipe": recipe})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Fungsi untuk mendapatkan foto terbaru berdasarkan userId dari MongoDB
async def get_latest_photo(user_id: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://be-rasadhana-245949327575.asia-southeast2.run.app/photos/latest/{user_id.strip()}")
            if response.status_code == 200:
                latest_photo_data = response.json()
                # # Debug output to check response format
                # print(f"Response from photo service: {latest_photo_data}")
                
                # Periksa apakah photoUrl ada di dalam respons 
                if 'photoUrl' in latest_photo_data['data']:
                    return latest_photo_data['data']  # Mengambil data foto dari response
                else:
                    raise HTTPException(status_code=404, detail="Foto tidak ditemukan atau 'photoUrl' tidak tersedia.")
            else:
                raise HTTPException(status_code=404, detail="Foto tidak ditemukan")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Mengambil PORT dari environment variable
    uvicorn.run(app, host="0.0.0.0", port=port)