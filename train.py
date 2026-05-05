import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# 1. Data configuration
DATA_PATH = "data/pokemon"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. Data generators for Training and Validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training')

val_gen = datagen.flow_from_directory(
    DATA_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation')

# 3. Function to build 4 different experimental models
def build_experiment(exp_no):
    if exp_no == 1: 
        # Exp 1: MobileNetV2 with frozen weights
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
        base.trainable = False
    elif exp_no == 2: 
        # Exp 2: MobileNetV2 with fine-tuning (last 20 layers)
        base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
        base.trainable = True
        for layer in base.layers[:-20]: layer.trainable = False
    elif exp_no == 3: 
        # Exp 3: ResNet50 with frozen weights
        base = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
        base.trainable = False
    elif exp_no == 4: 
        # Exp 4: ResNet50 with fine-tuning (last 20 layers)
        base = tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
        base.trainable = True
        for layer in base.layers[:-20]: layer.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
    return model

# 4. Loop to run and save all 4 experiments
experiments = [1, 2, 3, 4]
for i in experiments:
    print(f"\n--- Starting Experiment {i} ---")
    model = build_experiment(i)
    history = model.fit(train_gen, validation_data=val_gen, epochs=5)
    model.save(f"models/pokemon_exp_{i}.h5")
    print(f"Experiment {i} saved successfully!")