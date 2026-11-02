import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.convnext import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

IMG_SIZE = 224
BATCH_SIZE = 16
test_dir = "dataset/test"
model_path = "best_skin_convnext_phase1.keras"

print("🔍 1. กำลังเตรียมข้อมูล Test Data...")
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"\n📦 2. กำลังโหลดโมเดลจาก {model_path}...")
model = tf.keras.models.load_model(model_path, compile=False)

# ==========================================
# 3. คำนวณค่า Test Loss และ Test Accuracy รวม
# ==========================================
print("\n📊 3. กำลังประเมินผลรวม (Evaluate)...")
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
test_loss, test_acc = model.evaluate(test_data, verbose=1)

print("\n" + "="*50)
print(f"🎯 Test Accuracy (ความแม่นยำรวม): {test_acc * 100:.2f}%")
print(f"📉 Test Loss (ความผิดพลาดรวม):     {test_loss:.4f}")
print("="*50)

# ==========================================
# 4. ทำนายผลและแสดง Classification Report
# ==========================================
print("\n🚀 4. กำลังทำนายผลแยกรายโรค...")
y_pred_probs = model.predict(test_data, verbose=1)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

print("\n" + "="*65)
print("📋 CLASSIFICATION REPORT (ตารางความแม่นยำแยกรายโรค)")
print("="*65)
print(classification_report(y_true, y_pred_classes, target_names=class_names, digits=4))

# ==========================================
# 5. วาดและบันทึกรูป Confusion Matrix
# ==========================================
print("\n📊 5. กำลังสร้างรูปภาพ Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Class (คลาสจริง)', fontsize=12)
plt.xlabel('Predicted Class (คลาสที่ทำนาย)', fontsize=12)
plt.title(f'Confusion Matrix - ConvNeXt (Test Acc: {test_acc*100:.2f}%)', fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

output_cm = "confusion_matrix_convnext.png"
plt.savefig(output_cm, dpi=300)
print(f"✅ บันทึกรูป Confusion Matrix สำเร็จ: {output_cm}")
print("="*50)
plt.show()