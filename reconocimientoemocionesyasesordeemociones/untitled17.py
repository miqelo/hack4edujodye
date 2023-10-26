# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:27:59 2023

@author: ALMA ROSA
"""

import cv2
import os
import numpy as np
import time
import tkinter as tk

# Función para entrenar el modelo con un método específico
def train_model(method):
    emotion_recognizer = None

    if method == 'LBPH':
        emotion_recognizer = cv2.face_LBPHFaceRecognizer.create()

    if emotion_recognizer is not None:
        # Entrenando el reconocedor de emociones
        print("Entrenando (" + method + ")...")
        inicio = time.time()
        emotion_recognizer.train(facesData, np.array(labels))
        tiempoEntrenamiento = time.time() - inicio
        print("Tiempo de entrenamiento (" + method + "): ", tiempoEntrenamiento)

        # Almacenando el modelo obtenido
        emotion_recognizer.write("modelo" + method + ".xml")
        print("Modelo " + method + " almacenado con éxito.")

# Ruta donde se almacenarán los datos de emociones
dataPath = 'C:/Usuarios/ALMA ROSA/7 Reconocmiento de Emociones/Reconocimiento Emociones/Data'  # Ajusta la ruta según sea necesario

# Inicializa las listas para almacenar datos y etiquetas
labels = []
facesData = []
label = 0

for nameDir in os.listdir(dataPath):
    emotionsPath = os.path.join(dataPath, nameDir)

    for fileName in os.listdir(emotionsPath):
        labels.append(label)
        face_image = cv2.imread(os.path.join(emotionsPath, fileName), 0)
        facesData.append(face_image)

    label += 1

# Función para capturar emociones
def capture_emotion(emotion_name):
    emotion_path = os.path.join(dataPath, emotion_name)

    if not os.path.exists(emotion_path):
        os.makedirs(emotion_path)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret or count >= 200:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aux_frame = frame.copy()

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = aux_frame[y:y + h, x:x + w]
            face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(os.path.join(emotion_path, 'rostro_{}.jpg'.format(count)), face)
            count += 1

        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27 or count >= 200:
            break

    cap.release()
    cv2.destroyAllWindows()

# Crear ventana principal
root = tk.Tk()
root.title("Entrenamiento y Captura de Emociones")

# Botones para entrenar el modelo con diferentes métodos
train_lbph_button = tk.Button(root, text="Entrenar con LBPH", command=lambda: train_model('LBPH'))
train_lbph_button.pack()

# Botones para capturar emociones
happiness_button = tk.Button(root, text="Capturar Felicidad", command=lambda: capture_emotion("Felicidad"))
anger_button = tk.Button(root, text="Capturar Enojo", command=lambda: capture_emotion("Enojo"))
sadness_button = tk.Button(root, text="Capturar Tristeza", command=lambda: capture_emotion("Tristeza"))
surprise_button = tk.Button(root, text="Capturar Sorpresa", command=lambda: capture_emotion("Sorpresa"))

happiness_button.pack()
anger_button.pack()
sadness_button.pack()
surprise_button.pack()

# Iniciar la aplicación
root.mainloop()