import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Configuration ---
MODEL_PATH = "dog_cat_classifier.keras" # Ensure this path is correct
IMG_SIZE = (160, 160)
CLASS_NAMES = ['cat', 'dog'] 

# --- Model Loading ---
try:
    # Load the saved model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model. Check MODEL_PATH.")

# --- FastAPI App Setup ---
app = FastAPI(title="Dog vs Cat Classifier API")

# Configure CORS (Cross-Origin Resource Sharing)
# This is crucial for the frontend (index.html) running locally to communicate with the API.
origins = ["*"] # Allows all origins for development (Change in production!)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Response Model ---
class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: str
    raw_output: float

# --- Helper Function for Image Preprocessing ---
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Loads, resizes, and prepares the image for model prediction.
    """
    try:
        # Open the image from bytes
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize the image to the model's required input size
        img = img.resize(IMG_SIZE)
        
        # Convert to numpy array (0-255 range)
        img_array = np.array(img, dtype=np.float32)
        
        # Add a batch dimension (1, 160, 160, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # The model handles the rest of the preprocessing (like normalization)
        return img_array

    except Exception as e:
        print(f"Image preprocessing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file or format.")

# --- Prediction Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to receive an image file, process it, and return the prediction.
    """
    # 1. Read image file
    image_bytes = await file.read()
    
    # 2. Preprocess image
    image_tensor = preprocess_image(image_bytes)
    
    # 3. Make prediction
    # Prediction returns a single value between 0 and 1 (P(dog))
    raw_prediction = model.predict(image_tensor, verbose=0)[0][0] 
    
    # 4. Interpret result
    predicted_class_index = 1 if raw_prediction >= 0.5 else 0
    predicted_class_label = CLASS_NAMES[predicted_class_index]
    
    # 5. Determine confidence score
    confidence = float(raw_prediction) if predicted_class_index == 1 else float(1.0 - raw_prediction)
    
    # 6. Return response
    return {
        "filename": file.filename,
        "prediction": predicted_class_label,
        "confidence": f"{confidence * 100:.2f}%",
        "raw_output": float(raw_prediction)
    }

# --- Health Check Endpoint ---
@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": True}