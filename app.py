import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"} 

# Load the model architecture
with open('model_architecture.json', 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Load the model weights
model.load_weights('emotion_detection.weights.h5')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    
    if not ret:
        print(ret)
        
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    num_faces = face_detector.detectMultiScale(gray_frame, 1.3, 5)
    
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0)
        
        emotion_prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
cap.release()
cv2.destroyAllWindows()
        