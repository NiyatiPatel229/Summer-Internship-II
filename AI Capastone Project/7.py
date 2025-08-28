# --- Task 1: Load & summarize pre-trained CNN ---
from tensorflow.keras.models import load_model
model = load_model('model_path.h5')
model.summary()

# --- Task 2: Identify feature extraction layer ---
feature_layer_name = 'desired_layer_name'  # set according to your model

# --- Task 3: Define hybrid model ---
def build_cnn_vit_hybrid(cnn_base, vit_module):
    # Concatenate CNN and ViT outputs; customize as needed
    from tensorflow.keras.layers import Concatenate, Dense
    from tensorflow.keras.models import Model
    x = Concatenate()([cnn_base.output, vit_module.output])
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[cnn_base.input, vit_module.input], outputs=output)

# --- Task 4: Compile hybrid_model ---
hybrid_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Task 5: Set training configuration ---
batch_size = 16
epochs = 10
callbacks = [checkpoint_cb]

