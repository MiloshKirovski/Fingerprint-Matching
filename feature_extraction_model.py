import numpy as np
import cv2
import os
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def augment_image(image, num_augmentations=20):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    image = image.reshape((1,) + image.shape)
    augmented_images = []
    for _ in range(num_augmentations):
        for batch in datagen.flow(image, batch_size=1):
            augmented_images.append(batch[0].astype(np.float32))
            break
    return augmented_images


def load_images_from_folder(folder, target_size=(224, 224)):
    images = []
    for filename in glob(os.path.join(folder, '*.BMP')):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = np.stack([img] * 3, axis=-1)  # Convert grayscale to RGB
            img = img.astype(np.float32) / 255.0  # Normalize
            images.append(img)
    return np.array(images)


def prepare_data(images, num_classes):
    X = []
    y = []
    for i, img in enumerate(images):
        augmented = augment_image(img, num_augmentations=20)
        X.extend(augmented)
        y.extend([i] * len(augmented))
    return np.array(X), tf.keras.utils.to_categorical(y, num_classes=num_classes)


def create_simple_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


# Load and prepare data
images = load_images_from_folder('SIMPLE/REAL')
num_classes = len(images)
X, y = prepare_data(images, num_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and compile the model
input_shape = (224, 224, 3)
model = create_simple_model(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=10,
                    batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('fingerprint_recognition_model_10.h5')