import cv2
import numpy as np
from time import sleep#sleep for time to introduce delay 

#setting parameters

#for min height and width of the detected rectangle 
largura_min=80 
altura_min=80 

offset=6 #offset for pos. tolerance

pos_linha=550  #y_coordinates of the couting line 

delay= 60 #delay btw frames


detec = [] #detec is a list to store the centers of detected objects



carros= 0 # carros is a counter for the detected vehicles

#helper functions 

def pega_centro(x, y, w, h):#calculates the center coordinates of a rectangle
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

#used to capture frames from the specified video file
cap = cv2.VideoCapture('video.mp4')
#Background subtraction is performed to highlight moving objects.
subtracao =cv2.createBackgroundSubtractorMOG2()

# loop reads frames from the video, performs background subtraction, and processes the frames for vehicle detection.
while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    #This line calculates the time (tempo) to sleep before reading the next frame.
    sleep(tempo) 
    #used to pause the execution of the loop for the calculated time 

   #Image Processing 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((3,3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)

    #Contour Detection
    contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255,127,0), 3) 


    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)

    for (x,y) in detec:#check the center points in detec list
            if y<(pos_linha+offset) and y>(pos_linha-offset):
                carros+=1
                cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0,127,255), 3)  
                detec.remove((x,y))
                print("car is detected : "+str(carros))        
                
  
    cv2.putText(frame1, "VEHICLE COUNT : "+str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detectar",dilatada)
   #Exiting the Program
    if cv2.waitKey(1) == 27:
        break
#clean_up
cv2.destroyAllWindows()
cap.release()#Closes all OpenCV windows and releases the video capture object.

