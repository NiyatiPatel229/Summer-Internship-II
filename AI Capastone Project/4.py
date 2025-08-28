# --- Task 1: Create fnames list of all image files ---
import os
fnames = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            fnames.append(os.path.join(root, file))

# --- Task 2: Create validation_generator ---
from tensorflow.keras.preprocessing.image import ImageDataGenerator

val_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# --- Task 3: Count total CNN model layers ---
print("Number of layers:", len(model.layers))

# --- Task 4: Create and compile CNN ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    Conv2D(64, (3,3), activation='relu'),
    Conv2D(128, (3,3), activation='relu'),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Task 5: Define checkpoint callback for max accuracy ---
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_cb = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

# --- Task 6: Plot training & validation loss ---
import matplotlib.pyplot as plt
history = model.fit(...)  # your fit args here
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.show()

