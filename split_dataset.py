import os
import shutil
import random

# กำหนดโฟลเดอร์ต้นทาง และปลายทาง
SOURCE_DIR = "master_dataset"
DEST_DIR = "dataset"

# กำหนดสัดส่วนการแบ่ง (Train 70%, Validation 20%, Test 10%)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# กำหนดค่า Seed เพื่อให้การสุ่มมีความคงที่และตรวจสอบย้อนหลังได้
random.seed(42)

if not os.path.exists(SOURCE_DIR):
    print(f"ไม่พบโฟลเดอร์ต้นทาง: {SOURCE_DIR}")
    exit()

# สร้างโฟลเดอร์ dataset/train, val, test
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

# ดึงรายชื่อคลาสทั้งหมดจาก master_dataset
classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
classes.sort()

print(f"กำลังเริ่มคัดลอกและแบ่งข้อมูลจาก '{SOURCE_DIR}' ไปยัง '{DEST_DIR}'...")
print(f"สัดส่วน: Train {int(TRAIN_RATIO * 100)}% | Val {int(VAL_RATIO * 100)}% | Test {int(TEST_RATIO * 100)}%\n")

summary = []

for cls in classes:
    src_cls_dir = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # สุ่มสลับตำแหน่งภาพ
    random.shuffle(images)

    total = len(images)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    n_test = total - n_train - n_val  # ส่วนที่เหลือปัดเข้า test

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    # คัดลอกภาพไปยังแต่ละโฟลเดอร์
    for split_name, split_files in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        target_dir = os.path.join(DEST_DIR, split_name, cls)
        os.makedirs(target_dir, exist_ok=True)
        for img_name in split_files:
            shutil.copy2(os.path.join(src_cls_dir, img_name), os.path.join(target_dir, img_name))

    summary.append((cls, len(train_imgs), len(val_imgs), len(test_imgs), total))
    print(
        f"จัดการคลาส: {cls:<20} | Train: {len(train_imgs):<5} | Val: {len(val_imgs):<5} | Test: {len(test_imgs):<5} | รวม: {total}")

# แสดงตารางสรุปผลลัพธ์
print("\n" + "=" * 75)
print(f"{'Class Name':<25} | {'Train (70%)':<11} | {'Val (20%)':<9} | {'Test (10%)':<10} | {'Total':<6}")
print("=" * 75)
total_train = sum(s[1] for s in summary)
total_val = sum(s[2] for s in summary)
total_test = sum(s[3] for s in summary)
total_all = sum(s[4] for s in summary)

for s in summary:
    print(f"{s[0]:<25} | {s[1]:<11} | {s[2]:<9} | {s[3]:<10} | {s[4]:<6}")

print("=" * 75)
print(f"{'รวมทั้งหมด (Total)':<25} | {total_train:<11} | {total_val:<9} | {total_test:<10} | {total_all:<6}")
print("=" * 75)
print("\nแบ่งชุดข้อมูลเสร็จสมบูรณ์!")