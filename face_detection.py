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
    # tuple of x, y, w, h
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        
        #Detect the eyes 
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        
        #Detect eyes from the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_frame, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
    return frame

#Get frame from video
# 0 : system camera 
# 1 : external 
# 2 : external usb video
video_capter = cv2.VideoCapture(2)

while True:
    _,frame = video_capter.read()
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas  = detect(gray=gray,frame=frame)
    cv2.imshow('Video', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capter.release()
cv2.destroyAllWindows()
        
    


            
        
        
        
    
    
    