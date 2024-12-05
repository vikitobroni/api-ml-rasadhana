# Gunakan image Python yang ringan
FROM python:3.10-slim

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements untuk menginstall dependencies
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke dalam container
COPY . .

# Ekspose port yang digunakan oleh aplikasi
EXPOSE 8080

# Perintah untuk menjalankan aplikasi
CMD ["python", "main.py"]
