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

# ==========================================
# PHASE 2: FINE-TUNING
# ==========================================
print("\nStarting Phase 2: Fine-Tuning...")

# 1. Unfreeze the base model
base_model.trainable = True

# 2. Let's see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# 3. Fine-tune from this layer onwards (Leave the first 100 layers frozen)
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 4. Re-compile the model (CRITICAL: Use a much lower learning rate!)
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=1e-5),  # Tiny learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Continue training for a few more epochs
fine_tune_epochs = 5
total_epochs = 5 + fine_tune_epochs

model.fit(train_generator, 
          validation_data=val_generator, 
          epochs=total_epochs, 
          initial_epoch=5)  # Start from epoch 5
# ==========================================

# 4. Save Model
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")

# Save class names to a text file for the app to read
with open("classes.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")