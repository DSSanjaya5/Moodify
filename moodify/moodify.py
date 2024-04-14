import cv2
import numpy as np
import random
import gradio as gr
import keras.utils as image

from utils.model import Model
from utils.musicdataloader import MusicDataLoader


class Moodify:

    def __init__(self):
        self.model_loader = Model()
        self.music_loader = MusicDataLoader()
        self.loaded_model = self.model_loader.load_model()
        self.label_dict = self.model_loader.lables()
        self.happy_links, self.calm_links, self.sad_links, self.energetic_links = (
            self.music_loader.load_music_id()
        )
        self.happy, self.calm, self.sad, self.energetic = (
            self.music_loader.load_music_names()
        )

    def mood_detector(self, img, quality=0.8):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        print(gray.shape)
        face_haar_cascade = self.model_loader.face_classifier()
        faces = face_haar_cascade.detectMultiScale(gray, 1.3, 5)
        song_name = ""
        song = ""
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)

            predictions = self.loaded_model.predict(img_pixels)
            emotion_label = np.argmax(predictions)

            emotion_prediction = self.label_dict[emotion_label]

            cv2.putText(
                img,
                emotion_prediction,
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                1,
            )

            resize_image = cv2.resize(img, (1000, 700))
            # cv2.imshow("Img", resize_image)
            print("\n\n Emotion - ", emotion_prediction)
            if emotion_prediction == "Happiness" or emotion_prediction == "Surprise":
                song = random.choice(self.happy_links)
                song_name = self.happy[self.happy_links.index(song)]
            elif emotion_prediction == "Disgust" or emotion_prediction == "Angry":
                song = random.choice(self.energetic_links)
                song_name = self.energetic[self.energetic_links.index(song)]
            elif emotion_prediction == "Neutral":
                song = random.choice(self.calm_links)
                song_name = self.calm[self.calm_links.index(song)]
            elif emotion_prediction == "Sad" or emotion_prediction == "Fear":
                song = random.choice(self.sad_links)
                song_name = self.sad[self.sad_links.index(song)]
        print("\n\n Now Playing - ", song_name)
        return song, emotion_prediction

    def RunMoodify(self, img):
        song_id, emotion = self.mood_detector(img)
        vid = f"""<iframe width=100% height=490 max-width:100% src='https://www.youtube.com/embed/{song_id}?autoplay=1' title='Moodify Player' frameborder='0' allow='autoplay'></iframe>"""
        return gr.HTML(vid), emotion
