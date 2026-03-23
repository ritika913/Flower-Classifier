import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

DATASET_PATH = r'D:\CV_Project\flowers'
MODEL_NAME = 'flower_model.h5'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# 1. Prepare Data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Print the classes found
class_names = list(train_generator.class_indices.keys())
print("Classes found:", class_names)

# 2. Build Model (Transfer Learning)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the pre-trained weights

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Train
print("Starting training...")
model.fit(train_generator, validation_data=val_generator, epochs=5)

# 4. Save Model
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")

# Save class names to a text file for the app to read
with open("classes.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")