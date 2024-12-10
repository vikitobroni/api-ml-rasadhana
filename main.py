import os
import io
import cv2
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
import requests

# Load variabel lingkungan dari file .env
load_dotenv()

# Baca variabel dari file .env
MONGO_URL = os.getenv("MONGO_URL")
JWT_SECRET = os.getenv("JWT_SECRET")
EMAIL = os.getenv("EMAIL")
PASSWORD_EMAIL = os.getenv("PASSWORD_EMAIL")

# Koneksi ke MongoDB
client = MongoClient(MONGO_URL)
db = client["test"]  # Nama database
recipe_history_collection = db["RecipeHistory"]  # Koleksi untuk menyimpan riwayat resep
bookmark_collection = db["Bookmarks"]  # Koleksi untuk menyimpan bookmark resep

# Nama kelas
class_names = [
    'caberawit', 'tomat', 'wortel', 'tempe', 'bawangputih', 
    'dagingsapi', 'kentang', 'dagingayam', 'bawangmerah', 'telurayam'
]

# Fungsi untuk memuat model secara aman
def load_model_safe(model_path):
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"Model '{model_path}' berhasil dimuat.")
            return model
        else:
            raise FileNotFoundError(f"Model file '{model_path}' tidak ditemukan.")
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model: {e}")

# Memuat model klasifikasi gambar
model_path = "model_food_classification2.h5"
model = load_model_safe(model_path)

# Memuat model TF-IDF dan dataset untuk NLP
try:
    vectorizer = joblib.load('tfidf_vectorizer_model.sav')
    ingredient_vectors = joblib.load('ingredient_vectors.sav')
    data = pd.read_csv('cleaned_dataset.csv')
    print("Model NLP dan dataset berhasil dimuat.")
except Exception as e:
    raise RuntimeError(f"Gagal memuat model NLP atau dataset: {e}")

# Fungsi untuk merekomendasikan resep
def recommend_recipes(input_ingredients, top_n=8):
    input_vector = vectorizer.transform([input_ingredients])
    cosine_similarities = cosine_similarity(input_vector, ingredient_vectors).flatten()
    related_indices = cosine_similarities.argsort()[-top_n:][::-1]
    recommendations = data.iloc[related_indices]
    return recommendations[['Title', 'Ingredients', 'Steps']]

# Fungsi untuk memuat gambar dari URL
def load_image_from_url(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img_data = response.content
        img = Image.open(io.BytesIO(img_data))
        img = img.convert("RGB")  # Pastikan gambar dalam format RGB
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Gagal memuat gambar dari URL: {e}")

# Fungsi untuk memprediksi kelas dari gambar
def predict_single_image(model, img_data, target_size=(224, 224)):
    try:
        img_resized = cv2.resize(img_data, target_size)
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        return predicted_class, confidence
    except Exception as e:
        raise RuntimeError(f"Gagal memprediksi gambar: {e}")

# Fungsi untuk menyimpan riwayat resep ke MongoDB
def save_recipe_history(user_id, class_name, confidence, recipes):
    history_data = {
        "userId": user_id,
        "predictedIngredient": class_name,
        "confidence": confidence,
        "recipes": recipes
    }
    try:
        recipe_history_collection.insert_one(history_data)
        print("Riwayat resep berhasil disimpan ke MongoDB.")
    except Exception as e:
        raise RuntimeError(f"Gagal menyimpan riwayat resep: {e}")

# Inisialisasi aplikasi Flask
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Hello world from ML endpoint!"

@app.route("/predict_latest_image/<user_id>", methods=["POST"])
def predict_latest_image(user_id):
    try:
        # Bersihkan `user_id` dari karakter whitespace atau newline
        user_id = user_id.strip()

        # Ambil foto terbaru berdasarkan userId
        latest_photo_data = get_latest_photo(user_id)
        photo_url = latest_photo_data['photoUrl']  # Pastikan key 'photoUrl' ada

        # Baca gambar langsung dari URL
        photo_data = load_image_from_url(photo_url)

        # Prediksi dengan model
        predicted_class_id, confidence = predict_single_image(model, photo_data, target_size=(224, 224))
        class_name = class_names[predicted_class_id]

        # Konversi confidence dari float32 ke float
        confidence = float(confidence)

        # Gunakan NLP untuk merekomendasikan resep berdasarkan bahan yang diprediksi
        recommended_recipes = recommend_recipes(class_name)
        
        # Format hasil rekomendasi
        recipes = recommended_recipes.to_dict(orient='records')

        # Simpan riwayat ke MongoDB
        save_recipe_history(user_id, class_name, confidence, recipes)

        return jsonify({"class": class_name, "confidence": confidence * 100, "recipes": recipes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Fungsi untuk mendapatkan foto terbaru berdasarkan userId dari MongoDB
def get_latest_photo(user_id: str):
    try:
        response = requests.get(f"https://be-rasadhana-245949327575.asia-southeast2.run.app/photos/latest/{user_id.strip()}")
        response.raise_for_status()
        latest_photo_data = response.json()

        # Debug output to check response format
        print(f"Response from photo service: {latest_photo_data}")

        # Periksa apakah photoUrl ada di dalam respons 
        if 'photoUrl' in latest_photo_data['data']:
            return latest_photo_data['data']  # Mengambil data foto dari response
        else:
            raise ValueError("Foto tidak ditemukan atau 'photoUrl' tidak tersedia.")
    except Exception as e:
        raise ValueError(f"Error saat mendapatkan foto terbaru: {e}")

# Endpoint untuk menambahkan resep ke bookmark
@app.route("/bookmark_recipe", methods=["POST"])
def bookmark_recipe():
    try:
        # Ambil data dari request body
        data = request.json
        user_id = data.get("userId")
        recipe = data.get("recipe")

        # Validasi input
        if not user_id or not recipe:
            return jsonify({"error": "userId dan recipe harus disediakan."}), 400

        # Tambahkan informasi bookmark ke database
        bookmark_data = {
            "userId": user_id,
            "recipe": recipe
        }

        # Simpan data ke koleksi Bookmarks
        bookmark_collection.insert_one(bookmark_data)
        print("Resep berhasil ditambahkan ke bookmark.")

        return jsonify({"message": "Resep berhasil ditambahkan ke bookmark.", "bookmark": bookmark_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint untuk mengambil semua bookmark berdasarkan userId
@app.route("/get_bookmarks/<user_id>", methods=["GET"])
def get_bookmarks(user_id):
    try:
        # Cari semua bookmark milik user
        bookmarks = list(bookmark_collection.find({"userId": user_id}, {"_id": 0}))

        if not bookmarks:
            return jsonify({"message": "Tidak ada bookmark ditemukan untuk user ini."}), 404

        return jsonify({"bookmarks": bookmarks}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
