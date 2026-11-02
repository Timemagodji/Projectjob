import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 224
BATCH_SIZE = 16

train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# ==========================================
# 1. Data Generator & Data Augmentation
# ==========================================
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

# 💾 บันทึก class_indices.json อัตโนมัติ (เพื่อให้ app.py ดึงชื่อโรคไปใช้ได้ตรง 100%)
with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump(train_data.class_indices, f, indent=4, ensure_ascii=False)
print("\nอัปเดตไฟล์ class_indices.json เรียบร้อยแล้ว")

# ==========================================
# 2. คำนวณ Class Weights ปรับสมดุลข้อมูล
# ==========================================
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weight_dict = dict(enumerate(class_weights_array))

print("\nอัตราส่วน Class Weights ที่คำนวณได้:")
for class_idx, weight in class_weight_dict.items():
    class_name = list(train_data.class_indices.keys())[class_idx]
    print(f" - {class_name}: {weight:.2f}")
print("="*45)

# ==========================================
# 3. Attention Block (Channel Attention)
# ==========================================
def attention_block(inputs):
    channels = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(channels // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    out = layers.Multiply()([inputs, se])
    return out

# ==========================================
# 4. ประกอบโมเดล ResNet50 + Attention
# ==========================================
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

# ==========================================
# 5. ตั้งค่า Callbacks (บันทึกชื่อ Model3.keras)
# ==========================================
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
        'Model3.keras',  # 👈 ชื่อไฟล์โมเดลที่ดีที่สุด
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# ==========================================
# 6. Phase 1 Training (Feature Extraction)
# ==========================================
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

# ==========================================
# 7. Phase 2 Fine-Tuning
# ==========================================
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

# บันทึกโมเดลรอบสุดท้ายยืนยัน
# โหลดโมเดลรอบที่ดีที่สุดที่ ModelCheckpoint บันทึกไว้มาใช้ประเมินผล
print("\nกำลังโหลดโมเดลรอบที่ดีที่สุด (Best Weights) เพื่อประเมินผล...")
model = tf.keras.models.load_model('Model3.keras')
print(" โหลดโมเดล 'Model3.keras' สำเร็จ")

# ==========================================
# 8. Evaluation & วาดกราฟ 4 รูปตามเกณฑ์
# ==========================================
print("\nกำลังประเมินผลและสร้างกราฟรายงานผล...")
test_loss, test_acc = model.evaluate(test_data)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

# รวมค่าประวัติการเรียนรู้
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']
epochs_range = range(1, len(acc) + 1)
phase_split = len(history.history['accuracy'])

# -------------------------------------------------------------
# กราฟที่ 1: Loss_Curve.png
# -------------------------------------------------------------
plt.figure(figsize=(8.5, 5), dpi=300)
plt.plot(epochs_range, loss, label='Train Loss', color='#1f77b4', lw=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', color='#ff7f0e', lw=2)
plt.axvline(x=phase_split, color='red', linestyle='--', label='Start Fine Tuning')
plt.title('Loss Curve (Train / Validation Loss per Epoch)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Epochs', fontsize=11, fontweight='bold')
plt.ylabel('Loss', fontsize=11, fontweight='bold')
plt.legend(frameon=True, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Loss_Curve.png', dpi=300)
plt.close()
print("1/4 บันทึก: Loss_Curve.png สำเร็จ")

# -------------------------------------------------------------
# กราฟที่ 2: Accuracy_Curve.png
# -------------------------------------------------------------
plt.figure(figsize=(8.5, 5), dpi=300)
plt.plot(epochs_range, acc, label='Train Accuracy', color='#1f77b4', lw=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#ff7f0e', lw=2)
plt.axvline(x=phase_split, color='red', linestyle='--', label='Start Fine Tuning')
plt.title('Accuracy Curve (Train / Validation Accuracy per Epoch)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Epochs', fontsize=11, fontweight='bold')
plt.ylabel('Accuracy', fontsize=11, fontweight='bold')
plt.legend(frameon=True, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Accuracy_Curve.png', dpi=300)
plt.close()
print("2/4 บันทึก: Accuracy_Curve.png สำเร็จ")

# -------------------------------------------------------------
# กราฟที่ 3: PR_Curve.png
# -------------------------------------------------------------
test_data.reset()
y_pred_probs = model.predict(test_data)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

y_true_bin = label_binarize(y_true, classes=range(num_classes))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']

plt.figure(figsize=(9.5, 7), dpi=300)
for i in range(num_classes):
    p, r, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
    ap = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])
    plt.plot(r, p, lw=2, color=colors[i % len(colors)], label=f'{class_names[i]} (AP = {ap:.2f})')

p_micro, r_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_pred_probs.ravel())
mAP_micro = average_precision_score(y_true_bin, y_pred_probs, average='micro')
plt.plot(r_micro, p_micro, color='black', linestyle='--', lw=2.5, label=f'all classes mAP = {mAP_micro:.2f}')

plt.title('Precision-Recall (PR) Curve with mAP', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Recall', fontsize=11, fontweight='bold')
plt.ylabel('Precision', fontsize=11, fontweight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower left', fontsize=9, frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('PR_Curve.png', dpi=300)
plt.close()
print("3/4 บันทึก: PR_Curve.png สำเร็จ")

# -------------------------------------------------------------
# กราฟที่ 4: Confusion_Matrix.png
# -------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred_classes)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(9, 7.5), dpi=300)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix (Overall Test Accuracy: {test_acc * 100:.2f}%)', fontsize=13, fontweight='bold', pad=12)
plt.ylabel('True Class', fontsize=11, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9.5)
plt.yticks(rotation=0, fontsize=9.5)
plt.tight_layout()
plt.savefig('Confusion_Matrix.png', dpi=300)
plt.close()
print("4/4 บันทึก: Confusion_Matrix.png สำเร็จ")

print("\nเสร็จแล้วดีใจด้วย")