import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
from keras.models import model_from_json

class DataLoader:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def preprocess_data(self, test_size=0.1, val_size=0.1, random_state=42):
        emotions = self.data['emotion'].values.astype('int32')
        pixels = self.data['pixels'].values.tolist()
        pixels = np.array([np.fromstring(pixel, dtype='uint8', sep=' ') for pixel in pixels])
        pixels = pixels.reshape(-1, 48, 48, 1)
        emotions = utils.to_categorical(emotions)

        X_train, X_test, y_train, y_test = train_test_split(pixels, emotions, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def data_augmentation(self, X_train):
        datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')
        datagen.fit(X_train)
        return datagen

class CNNModel:
    def __init__(self):
        self.model = Sequential()

    def build_model(self):
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, datagen, X_train, y_train, X_val, y_val, batch_size=128, epochs=10):
        history = self.model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_val, y_val))
        return history

    def save_model(self, model_json_path, model_weights_path):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_weights_path)
        print("Saved model to disk")

    def load_model(self, model_json_path, model_weights_path):
        # load json and create model
        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(model_weights_path)
        print("Loaded model from disk")

    def evaluate_model(self, X_test, y_test):
        # Evaluate the model on the test set
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print('Test accuracy:', test_acc)

def main():
    # Load and preprocess data
    data_loader = DataLoader('/Moodify/fer2013.csv')
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.preprocess_data()
    datagen = data_loader.data_augmentation(X_train)

    # Build, compile, and train the model
    cnn_model = CNNModel()
    cnn_model.build_model()
    cnn_model.compile_model()
    history = cnn_model.train_model(datagen, X_train, y_train, X_val, y_val)

    # Save the model
    cnn_model.save_model("/Moodify/dataset/model100d.json", "/Moodify/dataset/model100d.h5")

if __name__ == "__main__":
    main()
