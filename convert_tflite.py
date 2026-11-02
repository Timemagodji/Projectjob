import tensorflow as tf
import json
import os

MODEL_KERAS_PATH = "Model3.keras"
TFLITE_OUTPUT_PATH = "Modelv3.tflite"
LABELS_OUTPUT_PATH = "labels.txt"

print(f"กำลังโหลดโมเดลจาก '{MODEL_KERAS_PATH}'...")
model = tf.keras.models.load_model(MODEL_KERAS_PATH)
print("โหลดโมเดลสำเร็จ!")

print("\nกำลังแปลงไฟล์เป็น TensorFlow Lite (.tflite)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# แปลงไฟล์โมเดล
tflite_model = converter.convert()

# บันทึกไฟล์ .tflite
with open(TFLITE_OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_OUTPUT_PATH) / (1024 * 1024)
print(f"บันทึกไฟล์โมเดลสำเร็จ: '{TFLITE_OUTPUT_PATH}' (ขนาด: {size_mb:.2f} MB)")

# 3. ส่งออกไฟล์ labels.txt สำหรับนำไปใส่ในโฟลเดอร์ assets ของ Flutter
print("\nกำลังสร้างไฟล์ 'labels.txt' สำหรับแอปพลิเคชันมือถือ...")
with open("class_indices.json", "r", encoding="utf-8") as f:
    class_indices = json.load(f)

# เรียงลำดับชื่อคลาสตาม index 0 ถึง 7
sorted_classes = sorted(class_indices.items(), key=lambda x: x[1])
labels = [name for name, idx in sorted_classes]

with open(LABELS_OUTPUT_PATH, "w", encoding="utf-8") as f:
    for name in labels:
        f.write(name + "\n")

print(f" บันทึกไฟล์ '{LABELS_OUTPUT_PATH}' สำเร็จ (มี {len(labels)} คลาส):")
for idx, name in enumerate(labels):
    print(f"   [{idx}] {name}")

print("\nเรียบร้อย")
print(f"   - {TFLITE_OUTPUT_PATH}")
print(f"   - {LABELS_OUTPUT_PATH}")