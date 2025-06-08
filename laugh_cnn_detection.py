import os
# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import glob  # For file path matching
import cv2  # OpenCV for image processing
from mtcnn.mtcnn import MTCNN  # MTCNN for face detection
import numpy as np  # Numpy for array manipulation
from sklearn.model_selection import train_test_split  # For data splitting
from joblib import dump  # To save the trained model
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras._tf_keras.keras.utils import to_categorical
from keras import models, layers, regularizers
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# Initialize lists for storing image data and corresponding labels
data = []
labels = []

# Function to detect faces in an image using MTCNN
def face_detector(img):
    try:
        detector = MTCNN()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_img)
        if faces:
            out = faces[0]
            x, y, w, h = out["box"]
            return img[y:y+h, x:x+w]  # Return the cropped face region
    except Exception as e:
        print(f"Error detecting face: {e}")
    return None  # If an error occurs or no face is detected

# Function to extract faces from a video
def extract_faces_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = MTCNN()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        for face in faces:
            x, y, w, h = face["box"]
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (32, 32))
            face_img = face_img.flatten() / 255.0  # Flatten and normalize
            data.append(face_img)
            labels.append("video_face")  # You can modify this label if needed
    cap.release()

# Check if preprocessed data already exists
if os.path.exists("preprocessed_data.npy") and os.path.exists("preprocessed_labels.npy"):
    print("[info] Loading preprocessed data...")
    data = np.load("preprocessed_data.npy")
    labels = np.load("preprocessed_labels.npy")
else:
    print("[info] Preprocessing data...")

    # Path to the dataset
    folder_path = Path(r"E:\machinlerning\laugh_cnn_camera_pichtcher_detection\smile_dataset")
    image_paths = glob.glob(str(folder_path / '**' / '*.[jp][pn]g'), recursive=True)

    # Process each image
    for i, item in enumerate(image_paths):
        img = cv2.imread(item)
        face = face_detector(img)
        if face is None:
            continue
        face = cv2.resize(face, (32, 32))
        face = face.flatten() / 255.0  # Flatten and normalize pixel values
        data.append(face)
        label = item.split("\\")[-2]  # Extract label from the file path
        labels.append(label)
        if i % 100 == 0:
            print(f"[info] : {i}/{len(image_paths)} processed")

    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    print("[info] Saving preprocessed data...")
    np.save("preprocessed_data.npy", data)
    np.save("preprocessed_labels.npy", labels)

# Extract faces from a video (if a video file is specified)
#video_file_path = r"E:\machinlerning\Laugh_detection\Q3\smile_dataset\video.mp4"  # Replace with your video file path
#extract_faces_from_video(video_file_path)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
def preprocess_data(data, labels):
    try:
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return x_train, x_test, y_train, y_test, label_encoder

    except Exception as e:
        print(f"[ERROR] : Error in data preprocessing: {e}")
        return None, None, None, None, None

# Function to build an optimized neural network model
def build_simple_model(input_shape, num_classes):
    try:
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(num_classes, activation='softmax')
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        return model

    except Exception as e:
        print(f"[ERROR] : Error building the model: {e}")
        return None


def plot_history(history):
    try:
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[ERROR] : Error plotting history: {e}")
x_train, x_test, y_train, y_test, label_encoder = preprocess_data(data, labels)

# Reshape the data for input to the model
x_train = np.reshape(x_train, (-1, 32, 32, 3))
x_test = np.reshape(x_test, (-1, 32, 32, 3))

# Build the model
input_shape = (32, 32, 3)
num_classes = y_train.shape[1]
model = build_simple_model(input_shape, num_classes)

# Training configuration
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Image data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Fit the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    validation_data=(x_test, y_test),
                    epochs=120,
                    callbacks=[early_stopping, reduce_lr])

# Plot training history
plot_history(history)

# Save the model
model.save("face_detection_model_cnn.h5")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
