from __future__ import annotations

import joblib
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml


@dataclass
class TrainResult:
    model_path: Path
    report: str


class StudentIdRecognizer:
    """Train and run a simple digit recognizer for student IDs.

    The recognizer uses an MLP classifier trained on MNIST digits. It expects
    input images to contain a single digit scaled to 28x28 pixels; callers are
    responsible for segmenting multi-digit IDs into individual digit crops.
    """

    def __init__(self, model_path: Path = Path("models/mnist_mlp.joblib")) -> None:
        self.model_path = model_path
        self.model: Pipeline | None = None

    def load_data(self):
        """Load the MNIST dataset and split it into train and test sets."""
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X = mnist["data"] / 255.0
        y = mnist["target"].astype(int)
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train(self) -> TrainResult:
        """Train the MLP classifier and persist it to disk."""
        X_train, X_test, y_train, y_test = self.load_data()

        pipeline = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        batch_size=256,
                        max_iter=30,
                        random_state=42,
                        verbose=False,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.model_path)
        self.model = pipeline

        return TrainResult(model_path=self.model_path, report=report)

    def load_model(self) -> None:
        """Load a previously trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file {self.model_path} not found. Train the model first."
            )
        self.model = joblib.load(self.model_path)

    def predict_digit(self, image: Image.Image) -> int:
        """Predict a single digit from a PIL image.

        The image is converted to grayscale and resized to 28x28 pixels. Higher
        level code should ensure the image contains only one digit.
        """
        if self.model is None:
            self.load_model()

        grayscale = image.convert("L").resize((28, 28))
        arr = np.array(grayscale).astype(np.float32).reshape(1, -1) / 255.0
        assert self.model is not None
        prediction = self.model.predict(arr)
        return int(prediction[0])

    def predict_id(self, digit_images: Iterable[Image.Image]) -> str:
        """Predict an entire student ID from a sequence of digit images."""
        digits: List[str] = [str(self.predict_digit(img)) for img in digit_images]
        return "".join(digits)


if __name__ == "__main__":
    recognizer = StudentIdRecognizer()
    result = recognizer.train()
    print("Model saved to", result.model_path)
    print("Evaluation:\n", result.report)
