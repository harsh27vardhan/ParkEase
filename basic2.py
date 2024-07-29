import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import pygame
pygame.init()
pygame.mixer.init()


model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('parking3.mp4')
# cap=cv2.VideoCapture(1) # If have to capture by webcam.

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
   




area1=[(163,198),(163,349),(329,350),(332,198)]
area2=[(660,349),(652,199),(440,204),(448,357)]
area3=[(682,199),(683,353),(885,345),(870,205)]




while True:    
    ret,frame = cap.read()
    if not ret:
        break
    
    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
   
    list1=[]
    list2=[]
    list3=[]
   
    
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c or 'bus' in c or 'truck' in c:
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2

       
      
            results1=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
            if results1>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list1.append(c)  

            results2=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
            if results2>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
               list2.append(c)  

            results3=cv2.pointPolygonTest(np.array(area3,np.int32),((cx,cy)),False)
            if results3>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list3.append(c)  

              
            
    
    a1=(len(list1))
    a2=(len(list2))
    a3=(len(list3))
  
    o=(a1+a2+a3)
    space=(3-o)
    print(space)

    if a1==1:
        cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('1'),(231,404),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        cv2.putText(frame,"Available space : "+str(space),(100,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('1'),(231,404),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        cv2.putText(frame,"Available space : "+str(space),(100,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)

    if a2==1:
        cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('2'),(563,388),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        cv2.putText(frame,"Available space : "+str(space),(100,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('2'),(563,388),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        cv2.putText(frame,"Available space : "+str(space),(100,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)

    if a3==1:
        cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('3'),(808,381),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        cv2.putText(frame,"Available space : "+str(space),(100,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('3'),(808,381),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        cv2.putText(frame,"Available space : "+str(space),(100,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
   
    
    

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1)&0xFF==ord('q'): # ExitCode
        break
cap.release()
cv2.destroyAllWindows()
#stream.stop()


