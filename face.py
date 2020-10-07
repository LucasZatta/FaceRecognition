import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import math

path = '/home/zatta/Área de Trabalho/Face Recog/Refs' #define the path to the file where the images are stores
files = os.listdir(path)							  #access a list of itens in the defined path
index = random.randrange(0, len(files))               #select a random image 
path_new = path+'/'+str(files[index])                 #defining the new path(path to the randomly selected image)
print("nome  "+path_new)
    
img = cv2.imread(path_new,cv2.IMREAD_COLOR)           #image object using cv2, we first open it as a colored image
imgtest1 = img.copy()                                 #create a copy of the color image
imgtest = cv2.cvtColor(imgtest1, cv2.COLOR_BGR2GRAY)  #create a new image, now in grayscale  


facecascade = cv2.CascadeClassifier('/home/zatta/Área de Trabalho/Face Recog/Frontal_face_haarscascade.xml')    #path to the xmls
eye_cascade = cv2.CascadeClassifier('/home/zatta/Área de Trabalho/Face Recog/Eyes_haarscascade.xml')

image = cv2.imread(path_new)                          #this is the image that we will use, because the cascade uses
                                                      #grayscale images, we create another image and convert it to 
                                                      #rgb  
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)		  	
image_eyes = image.copy()							  #another image, this will be used to draw the eyes rectangles


faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.2, minNeighbors=5) 
print('Total number of Faces found: '+str(len(faces)))
eye_count=0

for(x,y,w,h) in faces:
	#print('toaq')
	radius = (math.sqrt(w**2+h**2))/2.5
	face_detect = cv2.rectangle(image, (x,y), (x+w,y+h),(255,255,0),2)
	face_detect = cv2.circle(image, (x+int(w/2),y+int(h/2)),int(radius),(255,0,255),2)
	face_detect = cv2.circle(image, (x+int(w/2),y+int(h/2)),2,(0,255,255),2)
	face_detect = cv2.cvtColor(face_detect,cv2.COLOR_BGR2RGB)  

	plt.imshow(face_detect)
	plt.show()
	face_detect = cv2.cvtColor(face_detect,cv2.COLOR_BGR2RGB)  
	cv2.imwrite('/home/zatta/Área de Trabalho/Face Recog/Output/'+str(files[index])+'output.jpg',face_detect )

	eyes = eye_cascade.detectMultiScale(imgtest)
	print('Total eye candidates found :'+ str(len(eyes)))
	
	for (ex,ey,ew,eh) in eyes:
		print('x: '+str(x)+'              ex: '+str(ex))
		if(ex>x and ex<x+w):
			if(ey>y and ey<y+h):

				eye_detect = cv2.rectangle(image_eyes,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
				eye_detect = cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(255,0,255),2) 	#use this for rectangles eyes used in the same image as the face rectangle/circle
	    
				plt.imshow(eye_detect)
				eye_count+=1
				print('toaq')
				if(eye_count==(len(faces)*2)):
					break
plt.show()
eye_detect = cv2.cvtColor(face_detect,cv2.COLOR_BGR2RGB)  
cv2.imwrite('/home/zatta/Área de Trabalho/Face Recog/Output/'+str(files[index])+'output.jpg',eye_detect )
