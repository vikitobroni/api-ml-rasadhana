# Gunakan image Python sebagai base
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Salin semua file model dan dataset
COPY model_food_classification2.h5 tfidf_vectorizer_model.sav ingredient_vectors.sav cleaned_dataset.csv ./ 

# Salin requirements.txt dan instal dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . .

# Ekspos port 8080
EXPOSE 8080

# Jalankan aplikasi menggunakan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]



