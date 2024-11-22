import os
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
import shutil

# Ensure TensorFlow detects GPU (if available)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

folder_path = "/Users/ethanbonsall/Documents/Derm-Project/dataset"

try:
    shutil.rmtree(folder_path)
    print(f"Folder '{folder_path}' deleted successfully.")
except FileNotFoundError:
    print("Folder not found.")
except Exception as e:
    print(f"Error: {e}")

# Directories for the dataset
malignant_dir = "malignant_images"
non_malignant_dir = "non_malignant_images"
base_dir = "dataset"

# Combine malignant and non-malignant directories into a base directory
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    os.makedirs(os.path.join(base_dir, 'train/malignant'))
    os.makedirs(os.path.join(base_dir, 'train/non_malignant'))
    os.makedirs(os.path.join(base_dir, 'test/malignant'))
    os.makedirs(os.path.join(base_dir, 'test/non_malignant'))

# Function to copy files into train/test folders with oversampling
def split_and_copy_with_oversampling(src_dir, dest_train, dest_test, target_count, split_ratio=0.9):
    files = os.listdir(src_dir)
    train_size = int(len(files) * split_ratio)
    train_files = files[:train_size]
    test_files = files[train_size:]
    
    # Copy test files
    for file in test_files:
        dest_path = os.path.join(dest_test, file)
        tf.io.gfile.copy(os.path.join(src_dir, file), dest_path, overwrite=False)  # Overwrite enabled
    
    # Copy training files
    for file in train_files:
        dest_path = os.path.join(dest_train, file)
        tf.io.gfile.copy(os.path.join(src_dir, file), dest_path, overwrite=True)  # Overwrite enabled

    # Oversample the minority class by data augmentation if needed
    if len(train_files) < target_count:
        augment_data(dest_train, target_count - len(train_files))

# Function to augment data
def augment_data(directory, augment_count):
    datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Rotate images up to 20 degrees
    width_shift_range=0.2,  # Shift images horizontally up to 20% of the width
    height_shift_range=0.2,  # Shift images vertically up to 20% of the height
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Zoom in/out
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'  # Fill empty pixels after transformations
    )
    
    existing_files = os.listdir(directory)
    for i in range(augment_count):
        file_to_augment = random.choice(existing_files)
        img_path = os.path.join(directory, file_to_augment)
        
        # Load image and prepare for augmentation
        img = load_img(img_path)
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for generator
        
        # Generate augmented image
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=directory, save_prefix='aug', save_format='jpeg'):
            break  # Only one augmented image per loop

# Get counts for each class
malignant_count = len(os.listdir(malignant_dir))
non_malignant_count = len(os.listdir(non_malignant_dir))

# Determine the target count (maximum of the two classes)
target_train_count = max(malignant_count, non_malignant_count)

# Split datasets with oversampling for the minority class
split_and_copy_with_oversampling(malignant_dir, os.path.join(base_dir, 'train/malignant'), os.path.join(base_dir, 'test/malignant'), target_train_count)
split_and_copy_with_oversampling(non_malignant_dir, os.path.join(base_dir, 'train/non_malignant'), os.path.join(base_dir, 'test/non_malignant'), target_train_count)

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=(128, 128),  # Resize all images to 128x128
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Build the model
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),  # Dropout to prevent overfitting
    
    # Second Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),  # Dropout to prevent overfitting
    
    # Third Convolutional Block
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),  # Dropout to prevent overfitting
    
    
    # Flatten the 3D feature maps to 1D vector
    layers.Flatten(),
    
    # Dense Layers
    layers.Dense(512, activation='relu'),  # Increase Dense layer size
    layers.Dropout(0.5),  # Dropout for Dense layer to prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Binary classification output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(
    train_generator,
    epochs=40,  # Adjust as needed
    validation_data=test_generator
)

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Function to predict on a single image
def predict_image(image_path, model):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = img_array.reshape((1,) + img_array.shape)
    prediction = model.predict(img_array)
    return "Malignant" if prediction[0][0] > 0.5 else "Non-Malignant"

# Example usage
image_path = "/Users/ethanbonsall/Documents/Derm-Project/dataset/test/malignant/000001.png"
print(predict_image(image_path, model))
