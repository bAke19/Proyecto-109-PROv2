import numpy as np
import pyautogui
import imutils
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips =[8, 12, 16, 20]
thumb_tip= 4

while True:
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    results = hands.process(img)


    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Acceder a las marcas de referencia por su posición
            lm_list=[]
            for id ,lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Matriz para almacenar "True" o "False" si el dedo está doblado
            finger_fold_status =[]
            for tip in finger_tips:
                # Obteniendo las marcas de referencia de las posición de las puntas y dibujando un círculo azul
                x,y = int(lm_list[tip].x*w), int(lm_list[tip].y*h)
                cv2.circle(img, (x,y), 15, (255, 0, 0), cv2.FILLED)

                # Escribiendo una condición para verificar si el dedo está doblado, 
                # es decir, si el valor inicial de la punta del dedo es menor que 
                # la posición inicial del dedo, que es la marca de referencia interior.
                # Para el índice del dedo, si el dedo está doblado, cambiar el color a verde
                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x,y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

             # Verificar si todos los dedos están doblados
            if all(finger_fold_status):
                
                #  ESCRIBIR EL CÓDIGO AQUÍ
                image = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                cv2.imwrite('in_the_memory_to_disk.png', image)

                #Guardar directo en el disco
                pyautogui.screenshot('straight_to_disk.png')

                image = cv2.imread('straight_to_disk.png')
                cv2.imshow("Captura de Pantalla", imutils.resize(image, width=600))




            mp_draw.draw_landmarks(img, hand_landmark,
            mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2),
            mp_draw.DrawingSpec((0,255,0),4,2))
    

    cv2.imshow("Seguimiento de manos", img)
    cv2.waitKey(1)






