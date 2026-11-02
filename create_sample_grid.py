import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

plt.rcParams['font.family'] = 'Leelawadee UI'
plt.rcParams['axes.unicode_minus'] = False

dest_root = 'dataset_samples_for_presentation'

classes_info = [
    ("Melanoma", "มะเร็งผิวหนังเมลาโนมา"),
    ("BCC", "มะเร็งผิวหนังเบซัลเซลล์"),
    ("SCC", "มะเร็งผิวหนังชนิดสความัสเซลล์"),
    ("Actinic Keratosis", "ระยะก่อนมะเร็งผิวหนัง"),
    ("Melanocytic Nevi", "ไฝ หรือ ปานธรรมดา"),
    ("Keratosis", "โรคกระเนื้อ"),
    ("Psoriasis", "โรคสะเก็ดเงิน"),
    ("Tinea", "โรคเชื้อราผิวหนัง / กลาก")
]

# ฟังก์ชัน Center Crop ให้เป็นสี่เหลี่ยมจัตุรัสเหมือนที่เข้าโมเดล (224x224)
def crop_square(img):
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return img.crop((left, top, right, bottom)).resize((224, 224), Image.Resampling.LANCZOS)

# สร้าง Grid ขนาด 8 แถว x 4 คอลัมน์
fig, axes = plt.subplots(8, 4, figsize=(15, 21), dpi=300, facecolor='#F8FAFC')

for row_idx, (cname, thai_name) in enumerate(classes_info):
    cdir = os.path.join(dest_root, cname)
    files = sorted(os.listdir(cdir))[:4]
    
    for col_idx in range(4):
        ax = axes[row_idx, col_idx]
        ax.axis('off')
        
        if col_idx < len(files):
            img_path = os.path.join(cdir, files[col_idx])
            try:
                with Image.open(img_path) as raw_img:
                    sq_img = crop_square(raw_img.convert('RGB'))
                    ax.imshow(sq_img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        # ใส่กรอบบางๆ รอบรูป เพื่อความคมชัด
        rect = patches.Rectangle((0, 0), 223, 223, linewidth=1.5, edgecolor='#CBD5E1', facecolor='none')
        ax.add_patch(rect)

    # ใส่ชื่อโรคประจำแถว (ด้านซ้ายของรูปแรกในแต่ละแถว) ขยายขนาดให้อ่านง่าย ชัดเจนเป็นพิเศษ
    row_ax = axes[row_idx, 0]
    
    # บรรทัดบน: ชื่อภาษาอังกฤษ (เด่นชัด ขนาด 16.5)
    row_ax.text(-25, 95, cname,
                fontsize=16.5, fontweight='bold', color='#0F172A',
                va='bottom', ha='right')
    
    # บรรทัดล่าง: ชื่อภาษาไทย (ขนาด 13.5)
    row_ax.text(-25, 115, f"({thai_name})",
                fontsize=13.5, fontweight='semibold', color='#334155',
                va='top', ha='right')

# ปรับ Layout ให้สวยงามลงตัว
plt.subplots_adjust(left=0.25, right=0.97, top=0.94, bottom=0.03, hspace=0.32, wspace=0.12)

fig.suptitle("ตัวอย่างข้อมูลภาพถ่ายรอยโรคจริงที่ใช้ในการฝึกสอนโมเดล ResNet-50 (คลาสละ 4 ตัวอย่าง รวม 32 ภาพ)",
             fontsize=15.5, fontweight='bold', color='#0F172A', y=0.975)

grid_output = "dataset_8classes_samples_grid.png"
plt.savefig(grid_output, dpi=300, facecolor='#F8FAFC')
plt.close(fig)
print(f"Generated: {grid_output}")

# Copy to brain artifact directory
import shutil
artifact_dir = r"C:\Users\thams\.gemini\antigravity\brain\8deca88b-ece6-4c0c-abcf-441f2e118289"
dest_artifact = os.path.join(artifact_dir, "dataset_8classes_samples_grid.png")
shutil.copy2(grid_output, dest_artifact)
print(f"Copied to artifact: {dest_artifact}")
