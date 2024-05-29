import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *

from sklearn import metrics

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# download haarcascade_frontalface_default from here "https://github.com/opencv/opencv/tree/master/data/haarcascades"


def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


class EmotionDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Emotion Detector')
        self.master.geometry('800x600')
        self.master.configure(background='#2C3E50')

        self.facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = FacialExpressionModel("model.json", "model.weights.h5")

        self.EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        self.create_widgets()

    def create_widgets(self):
        self.heading = Label(self.master, text='Emotion Detector', pady=20, font=('Helvetica', 30, 'bold'))
        self.heading.configure(background='#2C3E50', foreground="#ECF0F1")
        self.heading.pack()

        self.upload_button = Button(self.master, text="Upload Image", command=self.upload_image, padx=20, pady=10)
        self.upload_button.configure(background="#3498DB", foreground='white', font=('Helvetica', 15, 'bold'))
        self.upload_button.pack(pady=30)

        self.sign_image = Label(self.master)
        self.sign_image.pack(pady=20)

        self.label1 = Label(self.master, background='#2C3E50', font=('Helvetica', 20, 'bold'))
        self.label1.pack(pady=10)

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            uploaded.thumbnail((400, 400))
            im = ImageTk.PhotoImage(uploaded)

            self.sign_image.configure(image=im)
            self.sign_image.image = im
            self.label1.configure(text='')

            self.detect_emotion(file_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def detect_emotion(self, file_path):
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.facec.detectMultiScale(gray_image, 1.3, 5)
        try:
            for (x, y, w, h) in faces:
                fc = gray_image[y:y+h, x:x+w]
                roi = cv2.resize(fc, (48, 48))
                pred = self.EMOTIONS_LIST[np.argmax(self.model.predict(roi[np.newaxis, :, :, np.newaxis]))]
            self.label1.configure(foreground="#ECF0F1", text=f'Predicted Emotion: {pred}')
        except:
            self.label1.configure(foreground="#ECF0F1", text="Unable to detect")


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()
