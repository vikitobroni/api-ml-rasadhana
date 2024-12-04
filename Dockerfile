FROM python:3.10.3-slim-buster

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8080

# Expose port
EXPOSE 8080

# Set entrypoint with uvicorn (replacing the original CMD)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
