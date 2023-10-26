import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import os
import imutils
from PIL import Image
import time
from tkinter import messagebox, ttk
import time
import numpy as np
result=0

def guardar_datos():
    personName = campo_nombre.get()
    dataPath = 'C:/rostros/'
    personPath = dataPath + '/' + personName
    
    if not os.path.exists(personPath):
        print('Carpeta creada: ',personPath)
        os.makedirs(personPath)
    
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    count = 1
    
    while True:
        ret, frame = cap.read()
        if ret == False: break
        frame =  imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()
    
        faces = faceClassif.detectMultiScale(gray,1.3,5)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/'+personName+'_0'+'{}.jpg'.format(count),rostro)
        
            
            count = count + 1
            
        cv2.imshow('Tomas a Color',frame)
    
        k =  cv2.waitKey(1)
        if k == 27 or count >= 100:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def entrenamiento():
    dataPath = 'C:/rostros'
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)

    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        print('Leyendo las imágenes')

        for fileName in os.listdir(personPath):
            print('Rostros: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
            image = cv2.imread(personPath+'/'+fileName,0)
            cv2.imshow('image',image)
            cv2.waitKey(10)
        label = label + 1

    print('labels= ',labels)
    print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
    print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

    # Métodos para entrenar el reconocedor
    face_recognizer = cv2.face.EigenFaceRecognizer_create()

    # Entrenando el reconocedor de rostros
    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))

    # Almacenando el modelo obtenido
    face_recognizer.write('modeloEigenFace.xml')
    print("Modelo almacenado...")

    cv2.destroyAllWindows()

def reconocimiento():

 dataPath = 'c:/rostros'
 imagePaths = os.listdir(dataPath)
 print('imagePaths=',imagePaths)

 face_recognizer = cv2.face.EigenFaceRecognizer_create()
 #face_recognizer = cv2.face.FisherFaceRecognizer_create()
 #face_recognizer = cv2.face.LBPHFaceRecognizer_create()

 # Leyendo el modelo
 face_recognizer.read('modeloEigenFace.xml')
 #face_recognizer.read('modeloFisherFace.xml')
 #face_recognizer.read('modeloLBPHFace.xml')

 cap = cv2.VideoCapture(0)
 cap.set(3,800) # set Width
 cap.set(4,600) # set Height

 #cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
 #cap = cv2.VideoCapture('Video.mp4')

 faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

 while True:
     time.sleep(.1)
     ret,frame = cap.read()
     if ret == False: break
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     auxFrame = gray.copy()

     faces = faceClassif.detectMultiScale(gray,1.3,5)

     for (x,y,w,h) in faces:
         rostro = auxFrame[y:y+h,x:x+w]
         rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
         result = face_recognizer.predict(rostro)

         #cv2.putText(frame,'{}'.format(result),(x,y-5),1,1,(173,216,230),1,cv2.LINE_AA)
         
         # EigenFaces
         if result[1] < 500:
             cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),3,1,(173,216,230),1,cv2.LINE_AA)
             cv2.circle(frame, (x,y),(x+w,y+h),(50,205,50),2)

 #            cv2.putText(gray,'{}'.format(imagePaths[result[0]]),(x,y-25),3,1.1,(0,255,0),1,cv2.LINE_AA)
 #            cv2.rectangle(gray, (x,y),(x+w,y+h),(0,255,0),2)
             
         else:
             cv2.putText(frame,'Desconocido',(x,y-25),3,0.8,(0,0,255),1,cv2.LINE_AA)
             cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,128),3)
             
             #cv2.putText(gray,'Desconocido',(x,y-25),3,0.8,(0,0,255),1,cv2.LINE_AA)
             #cv2.rectangle(gray, (x,y),(x+w,y+h),(0,0,255),2)


     cv2.imshow('frame',frame)
     #cv2.imshow('gray',gray)
     k = cv2.waitKey('q')
     if k == 27:
         break

 cap.release()
 cv2.destroyAllWindows()
    
root = tk.Tk()
root.config(width=800, height=600)
root.title("Menu Reconocimiento")

boton3 = ttk.Button(text="Entrenar modelo ^.^", command=entrenamiento)
boton3.place(x=400, y=480)

boton4 = ttk.Button(text="¡Reconocimiento!", command=reconocimiento)
boton4.place(x=600, y=480)

ventana = tk.Tk()

etiqueta_nombre = tk.Label(ventana, text="Nombre completo:")
etiqueta_nombre.pack()

campo_nombre = tk.Entry(ventana)
campo_nombre.pack()

boton_guardar = tk.Button(ventana, text="Guardar datos", command=guardar_datos)
boton_guardar.pack()

ventana.mainloop()

root.mainloop()