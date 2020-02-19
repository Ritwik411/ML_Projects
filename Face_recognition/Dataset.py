# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:23:00 2019

@author: Ritwik Gupta
"""

"""
This It has two simple commands
Face_ recognition- Recognise faces in a photograph or folder full for photographs.
face_detection - Find faces in a photograph or folder full for photographs.
For face recognition, first generate a feature set by taking few image of your face and create a directory with the name of person and save their face image.
Then train the data by using the Face_ recognition module.By Face_ recognition module the trained data is stored as pickle file (.pickle).
By using the trained pickle data, we can recognize face.
The main flow of face recognition is first to locate the face in the picture and the compare the picture with the trained data set.If the there is a match, it gives the recognized label.
"""

# Collection of Images for dataset
#Importing the required libraries
import cv2
import os
import numpy as np
import face_recognition
import pickle
import imutils.paths as paths
import matplotlib.pyplot as plt


#Specifying the path for images
path = "E:\Study Material\Machine Learning\Face_recognition\images"+'\\'
id1 = input('Enter Username')
#Using the haar cascade classifier
face_cascade = cv2.CascadeClassifier('E:\Study Material\Machine Learning\Face_recognition\haarcascade_frontalface_default.xml')
#Capturing images of face
"""
This part of the program is for capturing of live images.
"""
def capture(id1):
    num = 0
    # Importing html smaples for Face and Eyes
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Starting your Laptop Camera for recording
    cap = cv2.VideoCapture(0)
    while num <20:
        ret, img = cap.read()  # Getting image from the camera
        #ret would be 1 if python can read data else 0
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Converting Image to GrayScale
        
        # Detecting Faces from image using Face_Samples
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
        # Iterating for the dimentions of the Detected faces to draw rectangle
        for (x,y,w,h) in faces:
            
            # Writing the image to the specified location
            cv2.imwrite(path+str(id1)+ "\\" +str(num)+ ".jpg", gray[y:y+h, x:x+w])    
            
            # Creating Rectangle around Face with color Red and of width 2
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow('img',img) # Displays the Image with rectangles on Face
            num+=1
    cap.release()
    cv2.destroyAllWindows()
    

#Error detection
try:
    # Creating a target Directory
    os.mkdir(path+str(id1))
    print("Directory " , path+str(id1),  " Created ") 
    capture(id1)
except FileExistsError:
    print("Directory " , path+str(id1) ,  " already exists")

   


"""
# Read the image and convert to grayscale format
os.chdir("E:\Study Material\Machine Learning\Face_recognition\images\\"+id1)
for file in os.listdir():
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #calculate coordinates 
    faces = face_cascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    plt.imshow(img)

#Face encoding for each peron's face

all_face_encodings = {}
for file in os.listdir(path+id1):
    img1 = face_recognition.load_image_file(file)
    all_face_encodings[id1] = face_recognition.face_encodings(img1)[0]
os.chdir("E:\Study Material\Machine Learning\Face_recognition")
with open('encoding1.pickle', 'wb') as f:
    pickle.dump(all_face_encodings, f)


#Training the model

dataset = "E:\Study Material\Machine Learning\Face_recognition\images\\"# path of the data set 
module = "E:\Study Material\Machine Learning\Face_recognition\encodings\encoding1.pickle" # were u want to store the pickle file 

imagepaths = list(paths.list_images(dataset))
knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagepaths):
    print("[INFO] processing image {}/{}".format(i + 1,len(imagepaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	
    boxes = face_recognition.face_locations(rgb, model= "hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
       knownEncodings.append(encoding)
       knownNames.append(name)
       print("[INFO] serializing encodings...")
       data = {"encodings": knownEncodings, "names": knownNames}
       output = open(module, "wb") 
       pickle.dump(data, output)
       output.close()
      
#Using camera for live detection and recognition of face
def main():
    encoding = "E:\Study Material\Machine Learning\Face_recognition\encodings\encoding1.pickle"
    data = pickle.loads(open(encoding,'rb').read())
    #print(data)
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened :
        ret,frame = cap.read()
    else:
        ret = False
    while(ret):
        ret,frame = cap.read()
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #rbg = imutils.resize(frame,width = 400)
        r = frame.shape[1]/float(rgb.shape[1])
        boxes = face_recognition.face_locations(rgb, model= "hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        
        for encoding in encodings:
                matches = face_recognition.compare_faces(np.array(encoding),np.array(data["encodings"]))
                name = "Unknown"
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    
                    for i in matchedIdxs:
                                  name = data["names"][i]
                                  counts[name] = counts.get(name, 0) + 1
                                  name = max(counts, key=counts.get)
                names.append(name)
        for ((top, right, bottom, left), name) in zip(boxes, names):
          top = int(top * r)
          right = int(right * r)
          bottom = int(bottom * r) 
          left = int(left * r)
          cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
          y = top - 15 if top - 15 > 15 else top + 15
          cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break 
                                               
    cv2.destroyAllWindows()
    cap.release()
if __name__ == "__main__":
    main()
    if name!=id1:
        print("Sorry,your name does not match!")
    else:
        print("Welcome "+name+"!")

 """      
       
       
       
       
       
       
       
       
       