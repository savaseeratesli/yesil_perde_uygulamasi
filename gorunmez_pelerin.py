import cv2

import numpy as np

cam=cv2.VideoCapture(0)

#Bulunacak renk aralığı
#Mavi
lower=np.array([92,124,136])
upper=np.array([110,242,255])

_,background=cam.read()#Background foto

#Filtre
#Değerleri kendine göre ayarla
kernel=np.ones((3,3),np.uint8)#Görüntü içi
kernel2=np.ones((11,11),np.uint8)#Görüntü dışı
kernel3=np.ones((13,13),np.uint8)


while(cam.isOpened()):
    _,frame=cam.read()
    
    
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
     
    mask=cv2.inRange(hsv,lower,upper)
    
    #Morfolojik işlemler görüntüyü iyileştirmek için
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel2)
    mask=cv2.dilate(mask,kernel3,iterations=2)#Büyütme
    
    
    mask_not=cv2.bitwise_not(mask)
    
    bg=cv2.bitwise_and(background,background,mask=mask)
    fg=cv2.bitwise_and(frame,frame,mask=mask_not)
    
    dst=cv2.addWeighted(bg,1,fg,1,0)
    
    #Görüntüler yanyana olsun
    dst=np.hstack((frame,dst))#H yatay V dikey
    
     
    cv2.imshow("Orjinal",frame)
    cv2.imshow("Mask",mask)
    cv2.imshow("Dst",dst)
    
    if cv2.waitKey(1)&0xFF==ord("q"):
        break
    
cam.release()
cv2.destroyAllWindows()
    









