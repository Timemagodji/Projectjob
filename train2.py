import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- เพิ่ม Import สำหรับวาดกราฟและตาราง ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight # เพิ่มใหม่
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
# ใส่ Class Weight แบบอัตโนมัติ (ใหม่)
# =============================
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights_array))

print("\n⚖️ อัตราส่วน Class Weights ที่คำนวณได้:")
for class_idx, weight in class_weight_dict.items():
    class_name = list(train_data.class_indices.keys())[class_idx]
    print(f" - {class_name}: {weight:.2f}")
print("="*30)

# =============================
# สร้างบล็อก Attention (ใหม่)
# =============================
def attention_block(inputs):
    channels = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(channels // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    out = layers.Multiply()([inputs, se])
    return out

# =============================
# Load ResNet50 + Attention
# =============================
base_model = ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = attention_block(x)
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
    ),
    ModelCheckpoint(
        'skin_disease_model_v3_attention.keras', # เปลี่ยนชื่อไฟล์ให้เป็นเวอร์ชันใหม่
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
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

print("\n=== Phase 1 Training ===")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# =============================
# Phase 2 Fine-Tuning
# =============================
print("\n=== Phase 2 Fine-Tuning ===")

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
    class_weight=class_weight_dict
)

# =============================
# Evaluate
# =============================
test_loss, test_acc = model.evaluate(test_data)
print("\nTest Accuracy:", test_acc)

print("\nTraining Complete! Model saved as 'skin_disease_model_v3.keras'")

# =============================
# ส่วนวาดกราฟและตาราง
# =============================
print("\nGenerating Plots and Confusion Matrix...")

# 1. วาดกราฟ Model Accuracy และ Loss
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
plt.savefig('accuracy_loss_graph.png')
print("Saved accuracy_loss_graph.png")
plt.show()

# 2. วาดตาราง Confusion Matrix
y_pred_probs = model.predict(test_data)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.title('Confusion Matrix - ResNet50 + Attention')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")
plt.show()