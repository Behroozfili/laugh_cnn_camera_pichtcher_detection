import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import img_to_array
from mtcnn.mtcnn import MTCNN

# Load the trained model
model = load_model('face_detection_model_cnn.h5')

# Load label encoder (optional, if necessary)
# You can also save and load the label encoder to match the class labels
# Example: label_encoder = LabelEncoder()
# label_encoder.classes_ = np.load('classes.npy')

# Initialize the face detector (MTCNN)
detector = MTCNN()

# Function to predict if the face is smiling or not
def predict_smile(face, model):
    try:
        face = cv2.resize(face, (32, 32))  # Resize to 32x32 as expected by the model
        face = face.astype("float") / 255.0  # Normalize the pixel values
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)  # Expand dimensions to fit the model input shape

        # Predict the class (smile or no smile)
        prediction = model.predict(face)[0]  # Get prediction

        # Get the class with the highest probability
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]  # Confidence score
        return predicted_class, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, 0.0

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Label names (based on your dataset classes)
# If you used the label encoder, you can map the predictions to the original labels
labels = ["Not Smiling", "Smiling"]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces in the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    # Loop over the detected faces
    for face in faces:
        x, y, w, h = face["box"]
        face_img = frame[y:y+h, x:x+w]

        # Predict smile or not
        predicted_class, confidence = predict_smile(face_img, model)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the label with confidence
        label = f"{labels[predicted_class]}: {confidence:.2f}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show the frame
    cv2.imshow("Smile Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
