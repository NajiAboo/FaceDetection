import cv2

#Loading Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, frame):
    """
     Detect the face from the image 
     gray: gray image of the original image
     frame: original image
    """
    # detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    