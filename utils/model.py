import os
import cv2
from keras.models import Sequential, model_from_json


class Model:

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(current_dir, "..", "dataset", "model100d.json")
        self.weights_path = os.path.join(current_dir, "..", "dataset", "model100d.h5")
        self.classifier_path = os.path.join(
            current_dir, "..", "dataset", "haarcascade_frontalface_default.xml"
        )

    def load_model(self):
        # load json and create model
        with open(self.model_path, "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.weights_path)
        return loaded_model

    def face_classifier(self):
        face_haar_cascade = cv2.CascadeClassifier(self.classifier_path)
        return face_haar_cascade

    def lables(self):
        label_dict = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happiness",
            4: "Sad",
            5: "Surprise",
            6: "Neutral",
        }
        return label_dict
