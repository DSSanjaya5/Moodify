from keras.models import load_model
from keras.models import Sequential, model_from_json
import cv2


class Model:

    def load_model(self):
        # load json and create model
        with open("D:/Projects/Moodify/dataset/model100d.json", "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("D:/Projects/Moodify/dataset/model100d.h5")
        return loaded_model

    def face_classifier(self):
        face_haar_cascade = cv2.CascadeClassifier(
            "D:/Projects/Moodify/dataset/haarcascade_frontalface_default.xml"
        )
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
