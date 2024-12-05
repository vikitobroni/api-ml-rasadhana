# Gunakan image Python sebagai base
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Salin requirements.txt dan instal dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file aplikasi ke dalam container
COPY . .

# Ekspos port 8080
EXPOSE 8080

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
