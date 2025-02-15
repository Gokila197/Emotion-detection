import numpy as np
import cv2
from keras.models import model_from_json  # Corrected import
from keras.preprocessing.image import img_to_array

# Load the model from JSON file
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load the weights into the model
model.load_weights("emotiondetector.h5")

# Haarcascade XML for face detection
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Labels for the emotions
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preprocessing function to convert images for model input
def extract_features(image):
    feature = img_to_array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape to fit the model input
    return feature / 255.0  # Normalize the image

# Start video capture for real-time detection
webcam = cv2.VideoCapture(0)

while True:
    # Read frames from the webcam
    ret, im = webcam.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Extract the face from the grayscale frame
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))  # Resize to match the model input size

        # Preprocess the face for model input
        face_img = extract_features(face)

        # Predict emotion
        pred = model.predict(face_img)
        pred_label = labels[np.argmax(pred)]  # Get the predicted label

        # Draw a rectangle around the face
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Add the label on top of the face
        cv2.putText(im, pred_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("emotiondetector", im)

    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()