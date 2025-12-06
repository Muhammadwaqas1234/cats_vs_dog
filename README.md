This project demonstrates the deployment of a binary image classification model (Dog vs. Cat) trained using TensorFlow/Keras and served via a high-performance FastAPI web API. The predictions are consumed and displayed by a simple, professional web frontend using HTML, CSS, and JavaScript.

🚀 Key Technologies
Model Training: TensorFlow 2.x, Keras (using MobileNetV2 for Transfer Learning)

Backend API: FastAPI (Python)

Web Server: Uvicorn

Frontend: HTML, CSS, JavaScript (using Fetch API for asynchronous communication)

📂 Project Structure
Your project directory should be set up as follows:

dog-cat-classifier/
├── dog_cat_classifier.keras  <-- 💾 Trained Model File
├── app.py                    <--  FastAPI Backend Code
└── index.html                <-- Frontend Web Interface
⚙️ Setup and Installation
1. Model & Data
Ensure you have your trained Keras model, named dog_cat_classifier.keras, in the root directory. This is the model the FastAPI application will load.

2. Python Environment
You need Python 3.8+ installed. Set up your environment and install the required Python libraries:

Bash

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn tensorflow pillow python-multipart
▶️ Running the Application
Step 1: Start the Backend API (FastAPI)
Run the app.py file using the Uvicorn ASGI server. This will host the prediction endpoint.

Bash

uvicorn app:app --reload
The API will start running at http://127.0.0.1:8000. You can verify the prediction endpoint at http://127.0.0.1:8000/predict (or view the interactive documentation at http://127.0.0.1:8000/docs).

Step 2: Run the Frontend
With the API running, simply open the index.html file in your web browser.

Navigate to the directory in your file explorer.

Double-click index.html.

The JavaScript in the page will automatically connect to the FastAPI endpoint running on port 8000.

🧪 Usage and Testing
Select Image: Click the "Select Image File" button (or drag and drop a file) and choose a picture of a dog or a cat.

Preview: The image preview will appear, and the "Analyze Image" button will become active.

Predict: Click "Analyze Image".

Result: The button will show "Processing...", and upon completion, the Analysis Report will display the Classification (Dog or Cat) and the Confidence score.
