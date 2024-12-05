import os
import io
import cv2
import json
import httpx
import joblib
import pandas as pd
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv

# Nonaktifkan opsi oneDNN (jika diperlukan)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Nama kelas
class_names = ['caberawit', 'tomat', 'wortel', 'tempe', 'bawangputih', 'dagingsapi', 'kentang', 'dagingayam', 'bawang merah', 'telurayam']

# Memuat model klasifikasi gambar
model = tf.keras.models.load_model("model_food_classification2.h5")

# Memuat model TF-IDF dan dataset untuk NLP
vectorizer = joblib.load('tfidf_vectorizer_model.sav')
ingredient_vectors = joblib.load('ingredient_vectors.sav')
data = pd.read_csv('cleaned_dataset.csv')

# Fungsi untuk merekomendasikan resep

def recommend_recipes(input_ingredients, top_n=8):
    input_vector = vectorizer.transform([input_ingredients])
    cosine_similarities = cosine_similarity(input_vector, ingredient_vectors).flatten()
    related_indices = cosine_similarities.argsort()[-top_n:][::-1]
    recommendations = data.iloc[related_indices]
    return recommendations[['Title', 'Ingredients', 'Steps']]

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

# Fungsi untuk memprediksi kelas dari gambar
def predict_single_image(model, img_data, target_size=(224, 224)):
    img_resized = cv2.resize(img_data, target_size)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    return predicted_class, confidence

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

        # Gunakan NLP untuk merekomendasikan resep berdasarkan bahan yang diprediksi
        recommended_recipes = recommend_recipes(class_name)
        
        # Format hasil rekomendasi
        recipes = recommended_recipes.to_dict(orient='records')

        return JSONResponse(content={"class": class_name, "confidence": confidence * 100, "recipes": recipes})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Fungsi untuk mendapatkan foto terbaru berdasarkan userId dari MongoDB
async def get_latest_photo(user_id: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://be-rasadhana-245949327575.asia-southeast2.run.app/photos/latest/{user_id.strip()}")
            if response.status_code == 200:
                latest_photo_data = response.json()
                # Debug output to check response format
                print(f"Response from photo service: {latest_photo_data}")

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
    port = os.getenv("PORT", 8080)
    uvicorn.run(app, host='localhost', port=port)
