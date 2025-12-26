# Student ID Recognition

This repository provides a minimal baseline for recognizing student ID digits using the MNIST dataset. The `StudentIdRecognizer` trains an MLP classifier and predicts single digits or a sequence of digits that represent a full ID.

## Getting started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train the model and generate an evaluation report:

   ```bash
   python recognizer.py
   ```

   The trained model is saved to `models/mnist_mlp.joblib` and the MNIST classification report is printed to the console.

3. Predict a student ID from digit crops:

   ```python
   from PIL import Image
   from recognizer import StudentIdRecognizer

   recognizer = StudentIdRecognizer()
   recognizer.load_model()

   digit_images = [
       Image.open("digit0.png"),
       Image.open("digit1.png"),
       Image.open("digit2.png"),
       # ... repeat for each digit in the ID
   ]

   predicted_id = recognizer.predict_id(digit_images)
   print("Predicted ID:", predicted_id)
   ```

> Note: Multi-digit images must be segmented into 28x28 grayscale digit crops before calling `predict_id`.
