# ML-Driven Recipe Recommendation Service

This repository contains the code for a Flask-based machine learning service that performs food classification and recommends recipes. The service integrates image classification using TensorFlow and Natural Language Processing (NLP) techniques for ingredient-based recipe recommendation.

## Features

1. **Image Classification**

   - Classifies food images into categories such as `caberawit`, `tomat`, `wortel`, etc.
   - Utilizes a pre-trained TensorFlow model.

2. **Recipe Recommendation**

   - Recommends recipes based on the classified food category.
   - Uses a TF-IDF vectorizer and cosine similarity for ingredient matching.

3. **Integration with External Services**

   - Fetches the latest uploaded image for a user via an external API.

4. **Image URL Support**
   - Allows image input via a URL for classification.

---

## Installation

### Prerequisites

- Python 3.8+
- Pip
- Virtual environment (optional)

### Clone the Repository

```bash
git clone https://github.com/vikitobroni/api-ml-rasadhana.git
cd ml-api-rasadhana
```

### Set Up the Environment

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the Application

1. Set the required environment variables:

   - `PORT`: The port number for the Flask application (default is 8080).

2. Start the Flask server:

   ```bash
   python app.py
   ```

3. The application will be available at `http://your-ip:port`.

### API Endpoints

#### **`GET /`**

- **Description**: Returns a welcome message.
- **Response**:
  ```json
  {
    "message": "Hello world from ML endpoint!"
  }
  ```

#### **`POST /predict_latest_image/<user_id>`**

- **Description**: Predicts the class of the latest uploaded image for the given user ID and provides recipe recommendations.
- **Request**:
  - `user_id`: The unique ID of the user.
- **Response**:
  ```json
  {
    "class": "class_name",
    "confidence": confidence_score,
    "recipes": [
        {
            "Title": "Recipe Title",
            "Ingredients": "List of ingredients",
            "Steps": "Cooking steps"
        }
    ]
  }
  ```
- **Error Handling**: Returns an error message if the image or recipe processing fails.

---

## File Structure

- **`main.py`**: Main Flask application.
- **`model_food_classification2.h5`**: Pre-trained TensorFlow model for image classification.
- **`tfidf_vectorizer_model.sav`**: TF-IDF vectorizer for NLP.
- **`ingredient_vectors.sav`**: Precomputed ingredient vectors for recipe matching.
- **`cleaned_dataset.csv`**: Dataset containing recipes and ingredients.

---

## How It Works

### Image Classification

1. The input image is resized to 224x224.
2. The TensorFlow model predicts the food category.
3. Outputs the predicted class and confidence score.

### Recipe Recommendation

1. The predicted class is used as input for the TF-IDF vectorizer.
2. Cosine similarity is computed between the input and precomputed ingredient vectors.
3. The top-N matching recipes are retrieved from the dataset.

### External Image Service

- The `/predict_latest_image/<user_id>` endpoint retrieves the latest image URL for a given user ID from an external API.
- The image is downloaded and processed for classification.

---

## Dependencies

- Flask
- tensorflow
- joblib
- scikit-learn
- opencv-python
- pandas
- numpy
- requests
- pillow
- gunicorn

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Deployment

### Using Docker

1. Build the Docker image local:
   ```bash
   docker build -t ml-api-rasadhana .
   ```

2. Run the container:
   ```bash
   docker run -p 8080:8080 ml-api-rasadhana
   ```

### Using GCP Cloud Run

1. create artifact regristry:
   ```bash
   gcloud artifacts repositories create ml-api-rasadhana --repository-format=docker --location=asia-southeast2 --async
   ```

2. create builds:
   ```bash
   gcloud builds submit --tag asia-southeast2-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/ml-api-rasadhana/rasadhana:1.0.0
   ```

3. deploy ke cloud run:
   ```bash
   gcloud run deploy --image asia-southeast2-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/ml-api-rasadhana/rasadhana:1.0.0
   ```

---

## Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---
