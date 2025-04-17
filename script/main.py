import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_images_from_folders(root_folder):
    images = []
    labels = []
    image_paths = []  # Adicionado para armazenar os caminhos das imagens

    label_mapping = {'Baixo risco': 0, 'Alto risco': 1}

    for label_folder in os.listdir(root_folder):
        label_path = os.path.join(root_folder, label_folder)

        # Check if it's a directory before accessing label_mapping
        if os.path.isdir(label_path):
            label = label_mapping.get(label_folder, -1)
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(label)
                    image_paths.append(img_path)

    return images, labels, image_paths  # Retornar também os caminhos das imagens

# Choose between RGB (True) and grayscale (False)
use_rgb = True

# Path to the local dataset
data_path = r'.\dataset_vegetation_on_electrical_grid'

# Load images and labels
images, labels, image_paths = load_images_from_folders(data_path)

# Ensure all images have the same dimensions
if use_rgb:
    # Resize RGB images to a consistent size
    images = [cv2.resize(img, (128, 128)) for img in images]
    images = np.array(images)
else:
    # Convert to grayscale, resize, and normalize
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    images = [cv2.resize(img, (128, 128)) for img in images]
    images = np.array(images) / 255.0

# Splitting the data into training and testing sets
train_images, test_images, train_labels, test_labels, train_image_paths, test_image_paths = train_test_split(images, labels, image_paths, test_size=0.2, random_state=0)

# Choose between RGB and grayscale
if use_rgb:
    # Keep images in RGB
    train_images = np.array(train_images)
    test_images = np.array(test_images)
else:
    # Convert to grayscale
    train_images = np.array(train_images)
    test_images = np.array(test_images)

# Convert labels to numpy array
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Model configuration
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3 if use_rgb else 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test Accuracy: {test_acc}')

# Make predictions on the test set (get probabilities)
predictions_probs = model.predict(test_images)

# Convert probabilities to predicted labels (0 or 1)
predictions = (predictions_probs > 0.5).astype(int)

# Specify the directory where files will be saved
save_directory = './results/'

# Save the entire model (architecture + weights + optimizer state)
model_file = os.path.join(save_directory, 'CNN_model.keras') if use_rgb else os.path.join(save_directory, 'CNN_model_gray.keras')
weights_file = os.path.join(save_directory, 'CNN_weights.keras') if use_rgb else os.path.join(save_directory, 'CNN_weights_gray.keras')

# Save the predictions, test labels, test image paths
np.save(os.path.join(save_directory, 'predictions.npy'), predictions)
np.save(os.path.join(save_directory, 'test_labels.npy'), test_labels)
np.save(os.path.join(save_directory, 'test_image_paths.npy'), np.array(test_image_paths))  # Salvar os caminhos das imagens

# Save the entire model (architecture + weights + optimizer state)
model.save(model_file)
# Save only the weights
model.save_weights(weights_file)
