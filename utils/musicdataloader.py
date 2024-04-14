import os
import pandas as pd


class MusicDataLoader:

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        self.calm_xlsx = pd.read_excel(
            os.path.join(current_dir, "..", "dataset", "Calm.xlsx")
        )
        self.happy_xlsx = pd.read_excel(
            os.path.join(current_dir, "..", "dataset", "Happy.xlsx")
        )
        self.sad_xlsx = pd.read_excel(
            os.path.join(current_dir, "..", "dataset", "sad.xlsx")
        )
        self.energetic_xlsx = pd.read_excel(
            os.path.join(current_dir, "..", "dataset", "energetic.xlsx")
        )

    def load_music_names(self):
        happy = list(self.happy_xlsx["Song Name"])
        calm = list(self.calm_xlsx["Song Name"])
        sad = list(self.sad_xlsx["Song Name"])
        energetic = list(self.energetic_xlsx["Song Name"])
        return happy, calm, sad, energetic

    def load_music_id(self):
        happy_links = list(self.happy_xlsx["Id"])
        calm_links = list(self.calm_xlsx["Id"])
        sad_links = list(self.sad_xlsx["Id"])
        energetic_links = list(self.energetic_xlsx["Id"])
        return happy_links, calm_links, sad_links, energetic_links
