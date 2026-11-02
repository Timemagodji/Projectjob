import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# --- เพิ่ม Import สำหรับวาดกราฟและตาราง ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# --------------------------------------

IMG_SIZE = 224
BATCH_SIZE = 16

train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# =============================
# Data Generator
# =============================
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_data.num_classes

# =============================
# ใส่ Class Weight
# =============================
class_weight = {
    0: 1.16,  # BCC
    1: 1.40,  # Keratosis
    2: 0.47,  # Melanocytic Nevi
    3: 0.47,  # Melanoma
    4: 1.87,  # Psoriasis
    5: 2.12,  # SCC
    6: 2.01,  # Tinea
    7: 3.37   # actinic keratosis
}

# =============================
# Load ResNet50
# =============================
base_model = ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# =============================
# Callbacks
# =============================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        patience=3,
        factor=0.3
    )
]

# =============================
# Phase 1 Training
# =============================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("=== Phase 1 Training ===")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weight
)

# =============================
# Phase 2 Fine-Tuning
# =============================
print("=== Phase 2 Fine-Tuning ===")

base_model.trainable = True

for layer in base_model.layers[:120]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weight
)

# =============================
# Evaluate
# =============================
test_loss, test_acc = model.evaluate(test_data)
print("Test Accuracy:", test_acc)

model.save("skin_disease_model_v2.keras")#เซฟโมเดล

print("Training Complete!")

# =============================
# ส่วนที่เพิ่มใหม่: วาดกราฟและตาราง
# =============================
print("Generating Plots and Confusion Matrix...")

# 1. วาดกราฟ Model Accuracy และ Loss
# รวมประวัติการเทรนจาก Phase 1 และ 2
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(14, 5))

# กราฟ Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Train Accuracy', marker='o', markersize=4)
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o', markersize=4)
plt.axvline(x=len(history.history['accuracy'])-1, color='r', linestyle='--', label='Start Fine Tuning')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# กราฟ Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Train Loss', marker='o', markersize=4)
plt.plot(epochs, val_loss, label='Validation Loss', marker='o', markersize=4)
plt.axvline(x=len(history.history['loss'])-1, color='r', linestyle='--', label='Start Fine Tuning')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('accuracy_loss_graph.png') # บันทึกเป็นไฟล์รูป
print("Saved accuracy_loss_graph.png")
plt.show()

# 2. วาดตาราง Confusion Matrix
# ให้โมเดลทำนายจากชุดข้อมูลทดสอบ
y_pred_probs = model.predict(test_data)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# ดึงเฉลยจริงและชื่อคลาสจากชุดข้อมูลทดสอบ
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

# สร้าง Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Class (เฉลยจริง)')
plt.xlabel('Predicted Class (โมเดลทาย)')
plt.title('Confusion Matrix - Skin Disease Model v2')
plt.xticks(rotation=45, ha='right') # หมุนแกน x ให้อ่านง่ายขึ้น
plt.tight_layout()
plt.savefig('1.png') # บันทึกเป็นไฟล์รูป
print("1.png")
plt.show()