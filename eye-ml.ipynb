# ============================================================
# 1. IMPORTS
# ============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 2. AUTO-DETECT DATASET PATH (ALREADY WORKING)
# ============================================================

BASE_INPUT = "/kaggle/input/kermany2018"

def find_oct_root(base_path):
    for root, dirs, files in os.walk(base_path):
        if set(["train", "test", "val"]).issubset(set(dirs)):
            return root
    raise FileNotFoundError("Could not find OCT dataset")

DATA_DIR = find_oct_root(BASE_INPUT)

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

print("✅ Dataset detected at:", DATA_DIR)
print("Train folders:", os.listdir(TRAIN_DIR))


# ============================================================
# 3. SETTINGS
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10


# ============================================================
# 4. DATA GENERATORS
# ============================================================

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_gen  = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

CLASS_NAMES = list(train_data.class_indices.keys())
print("Classes:", CLASS_NAMES)


# ============================================================
# 5. MODEL — RESNET50 (OFFLINE-SAFE TRANSFER LEARNING)
# ============================================================

print("\n🧠 Building ResNet50 model (offline-safe)...")

try:
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    print("✅ ImageNet weights loaded from cache")
except:
    print("⚠️ ImageNet weights not available (offline). Using random init.")
    base_model = ResNet50(
        include_top=False,
        weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(CLASS_NAMES), activation="softmax")
])

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# 6. TRAINING
# ============================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)


# ============================================================
# 7. EVALUATION
# ============================================================

test_data.reset()
preds = model.predict(test_data)
y_pred = np.argmax(preds, axis=1)
y_true = test_data.classes

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.show()


# ============================================================
# 8. GRAD-CAM (EXPLAINABLE AI)
# ============================================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def show_gradcam(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0

    preds = model.predict(np.expand_dims(img_norm, axis=0))
    cls = np.argmax(preds)
    conf = preds[0][cls] * 100

    heatmap = make_gradcam_heatmap(
        np.expand_dims(img_norm, axis=0),
        model,
        last_conv_layer_name="conv5_block3_out"
    )

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.title("Original OCT Image")
    plt.imshow(img_resized)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title(f"Prediction: {CLASS_NAMES[cls]}\nConfidence: {conf:.2f}%")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()


# ============================================================
# 9. EXAMPLE (OPTIONAL)
# ============================================================

# show_gradcam(os.path.join(TEST_DIR, "CNV", os.listdir(os.path.join(TEST_DIR, "CNV"))[0]))

