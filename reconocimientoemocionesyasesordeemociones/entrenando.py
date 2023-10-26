import cv2
import os
import numpy as np
import time
import tkinter as tk


def obtenerModelo(method,facesData,labels):

	if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

	# Entrenando el reconocedor de rostros
	print("Entrenando ( "+method+" )...")
	inicio = time.time()
	emotion_recognizer.train(facesData, np.array(labels))
	tiempoEntrenamiento = time.time()-inicio
	print("Tiempo de entrenamiento ( "+method+" ): ", tiempoEntrenamiento)

	# Almacenando el modelo obtenido
	emotion_recognizer.write("modelo"+method+".xml")

dataPath = 'C:/Usuarios/ALMA ROSA/7 Reconocmiento de Emociones/Reconocimiento Emociones/Data' #Cambia a la ruta donde hayas almacenado Data
emotionsList = os.listdir(dataPath)
print('Lista de personas: ', emotionsList)

labels = []
facesData = []
label = 0

for nameDir in emotionsList:
	emotionsPath = dataPath + '/' + nameDir

	for fileName in os.listdir(emotionsPath):
		#print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(emotionsPath+'/'+fileName,0))
		#image = cv2.imread(emotionsPath+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1


obtenerModelo('LBPH',facesData,labels)
