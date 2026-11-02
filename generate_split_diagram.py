import matplotlib.pyplot as plt
import matplotlib.patches as patches

# กำหนดฟอนต์ภาษาไทยมาตรฐานบน Windows
plt.rcParams['font.family'] = 'Leelawadee UI'
plt.rcParams['axes.unicode_minus'] = False

# สร้าง figure ขนาด 16:9 สำหรับสไลด์และการพิมพ์ (ความละเอียดสูง 300 DPI)
fig = plt.figure(figsize=(16, 9), dpi=300, facecolor='#F8FAFC')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# ==========================================
# 1. Header Title
# ==========================================
header_rect = patches.FancyBboxPatch(
    (0.8, 8.05), 14.4, 0.75,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#0F172A", edgecolor="none", zorder=2
)
ax.add_patch(header_rect)

ax.text(8.0, 8.52, "แผนผังการจัดสรรชุดข้อมูลและการฝึกสอนโมเดล ResNet-50",
        color="#FFFFFF", fontsize=18, fontweight='bold', ha='center', va='center', zorder=3)
ax.text(8.0, 8.22, "Dataset Stratified Splitting (70% : 20% : 10%) & Two-Phase Training Workflow",
        color="#94A3B8", fontsize=11, ha='center', va='center', zorder=3)

# ==========================================
# 2. Main Dataset Card (ซ้ายบน)
# ==========================================
ds_box = patches.FancyBboxPatch(
    (0.8, 6.15), 3.3, 1.6,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5, zorder=2
)
ax.add_patch(ds_box)

ds_bar = patches.FancyBboxPatch(
    (0.8, 7.35), 3.3, 0.4,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#0284C7", edgecolor="none", zorder=3
)
ax.add_patch(ds_bar)
ax.text(2.45, 7.55, "ชุดข้อมูลภาพรอยโรคผิวหนัง", color="#FFFFFF", fontsize=12, fontweight='bold', ha='center', va='center', zorder=4)

ax.text(2.45, 6.95, "33,619 ภาพ", color="#0F172A", fontsize=18, fontweight='bold', ha='center', va='center', zorder=4)
ax.text(2.45, 6.55, "• รวบรวมจาก ISIC Archive และ Kaggle\n• ครอบคลุมกลุ่มโรคผิวหนัง 8 คลาส", 
        color="#475569", fontsize=9.5, ha='center', va='center', zorder=4, linespacing=1.3)

# ==========================================
# 3. Stratified Splitting Hub (ตัวกระจายข้อมูล)
# ==========================================
split_box = patches.FancyBboxPatch(
    (4.6, 6.35), 2.2, 1.2,
    boxstyle="round,pad=0.08,rounding_size=0.18",
    facecolor="#EEF2FF", edgecolor="#4F46E5", linewidth=2, linestyle='--', zorder=2
)
ax.add_patch(split_box)
ax.text(5.7, 7.15, "Stratified Split", color="#3730A3", fontsize=12, fontweight='bold', ha='center', va='center', zorder=4)
ax.text(5.7, 6.72, "สุ่มแบบรักษาสัดส่วนทุกโรค\n70% : 20% : 10%", color="#1E293B", fontsize=9.5, ha='center', va='center', zorder=4, linespacing=1.2)

# ลูกศรจาก Dataset -> Split
ax.annotate('', xy=(4.55, 6.95), xytext=(4.15, 6.95),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2, mutation_scale=15), zorder=5)

# จุดรวมและเส้นทางแยกสัญญาณ Split (Routing Branch)
split_out_x = 6.85
# เส้นแยกแนวตั้ง
ax.plot([7.15, 7.15], [2.5, 6.9], color="#6366F1", lw=2.2, linestyle='-', zorder=4)
# เส้นเชื่อมจากกล่อง Split ไปเส้นแนวตั้ง
ax.plot([split_out_x, 7.15], [6.9, 6.9], color="#6366F1", lw=2.2, linestyle='-', zorder=4)

# ==========================================
# 4. สามกิ่ง: Train / Val / Test Cards (กลาง)
# ==========================================

# --- กิ่งที่ 1: Train Set (บน) ---
train_box = patches.FancyBboxPatch(
    (7.5, 6.0), 3.7, 1.8,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.8, zorder=2
)
ax.add_patch(train_box)
train_hdr = patches.FancyBboxPatch(
    (7.5, 7.4), 3.7, 0.4,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#1D4ED8", edgecolor="none", zorder=3
)
ax.add_patch(train_hdr)
ax.text(9.35, 7.6, "1. ชุดข้อมูลฝึกสอน : Train Set (70%)", color="#FFFFFF", fontsize=11.5, fontweight='bold', ha='center', va='center', zorder=4)
ax.text(9.35, 7.05, "23,531 ภาพ", color="#1E3A8A", fontsize=17, fontweight='bold', ha='center', va='center', zorder=4)
ax.text(9.35, 6.45, "• ปรับแต่งภาพด้วย Data Augmentation\n  (สุ่มหมุน 20°, ซูม 20%, พลิกแนวนอน)\n• คำนวณ Balanced Class Weights ลดความเอนเอียง\n• ป้อนให้โมเดลใช้เรียนรู้และปรับค่าน้ำหนัก", 
        color="#1E40AF", fontsize=9, ha='center', va='center', zorder=4, linespacing=1.25)

# --- กิ่งที่ 2: Validation Set (กลาง) ---
val_box = patches.FancyBboxPatch(
    (7.5, 3.8), 3.7, 1.8,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#FFFBEB", edgecolor="#D97706", linewidth=1.8, zorder=2
)
ax.add_patch(val_box)
val_hdr = patches.FancyBboxPatch(
    (7.5, 5.2), 3.7, 0.4,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#B45309", edgecolor="none", zorder=3
)
ax.add_patch(val_hdr)
ax.text(9.35, 5.4, "2. ชุดตรวจสอบ : Validation Set (20%)", color="#FFFFFF", fontsize=11.5, fontweight='bold', ha='center', va='center', zorder=4)
ax.text(9.35, 4.85, "6,721 ภาพ", color="#78350F", fontsize=17, fontweight='bold', ha='center', va='center', zorder=4)
ax.text(9.35, 4.25, "• ใช้ตรวจสอบความแม่นยำหลังจบแต่ละ Epoch\n• ป้องกันและเฝ้าระวังปัญหา Overfitting\n• ควบคุม EarlyStopping (หยุดเมื่อไม่พัฒนา 7 รอบ)\n• สั่งลด Learning Rate ด้วย ReduceLROnPlateau", 
        color="#92400E", fontsize=9, ha='center', va='center', zorder=4, linespacing=1.25)

# --- กิ่งที่ 3: Test Set (ล่าง) ---
test_box = patches.FancyBboxPatch(
    (7.5, 1.6), 3.7, 1.8,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.8, zorder=2
)
ax.add_patch(test_box)
test_hdr = patches.FancyBboxPatch(
    (7.5, 3.0), 3.7, 0.4,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#047857", edgecolor="none", zorder=3
)
ax.add_patch(test_hdr)
ax.text(9.35, 3.2, "3. ชุดทดสอบ : Test Set (10%)", color="#FFFFFF", fontsize=11.5, fontweight='bold', ha='center', va='center', zorder=4)
ax.text(9.35, 2.65, "3,367 ภาพ", color="#064E3B", fontsize=17, fontweight='bold', ha='center', va='center', zorder=4)
ax.text(9.35, 2.05, "• ภาพที่โมเดลไม่เคยพบมาก่อน (Unseen Data)\n• วัดผลชี้วัดทางการ: Test Accuracy 81.05%\n• ประเมิน Confusion Matrix, Precision, Recall\n• คำนวณกราฟ PR Curve (mAP รวม 90%)", 
        color="#065F46", fontsize=9, ha='center', va='center', zorder=4, linespacing=1.25)

# ลูกศรชี้เข้า Train, Val, Test จากเส้นแนวดิ่ง
ax.annotate('', xy=(7.45, 6.9), xytext=(7.15, 6.9),
            arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=2.2, mutation_scale=14), zorder=5)
ax.annotate('', xy=(7.45, 4.7), xytext=(7.15, 4.7),
            arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=2.2, mutation_scale=14), zorder=5)
ax.annotate('', xy=(7.45, 2.5), xytext=(7.15, 2.5),
            arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2.2, mutation_scale=14), zorder=5)

# ==========================================
# 5. โมเดล ResNet-50 Pipeline Box (ขวาสุด)
# ==========================================
model_box = patches.FancyBboxPatch(
    (11.8, 1.6), 3.4, 6.2,
    boxstyle="round,pad=0.08,rounding_size=0.18",
    facecolor="#FFFFFF", edgecolor="#4F46E5", linewidth=2, zorder=2
)
ax.add_patch(model_box)

model_hdr = patches.FancyBboxPatch(
    (11.8, 7.35), 3.4, 0.45,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#4338CA", edgecolor="none", zorder=3
)
ax.add_patch(model_hdr)
ax.text(13.5, 7.57, "สถาปัตยกรรมโมเดล ResNet-50", color="#FFFFFF", fontsize=11.5, fontweight='bold', ha='center', va='center', zorder=4)

ax.text(13.5, 7.05, "ขนาดภาพนำเข้า: 224 × 224 × 3", color="#0F172A", fontsize=9.5, fontweight='bold', ha='center', va='center', zorder=4)

blocks = [
    ("Pre-trained ResNet-50 Backbone\n(โครงสร้างหลัก 50 เลเยอร์ จาก ImageNet)", "#EEF2FF", "#3730A3"),
    ("+ SE-Attention Block\n(Channel Attention เน้นฟีเจอร์สำคัญของรอยโรค)", "#FEF3C7", "#92400E"),
    ("Classifier Head\n(GlobalAvgPool + BatchNorm + Dense 256 + Dropout 0.5)", "#F3E8FF", "#6B21A8"),
    ("Output Layer (Softmax)\n(จำแนก 8 คลาส พร้อมค่าความน่าจะเป็น 0-100%)", "#ECFDF5", "#065F46"),
    ("กลยุทธ์การฝึกสอนแบบ 2 ระยะ (Two-Phase)\n• Phase 1: Feature Extraction (LR = 1e-3)\n• Phase 2: Fine-Tuning เลเยอร์ >120 (LR = 1e-5)", "#F8FAFC", "#334155")
]

cur_y = 6.45
for text, bg, border in blocks:
    h = 0.76 if "Phase" in text else 0.62
    b = patches.FancyBboxPatch(
        (12.0, cur_y - h), 3.0, h,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor=bg, edgecolor=border, linewidth=1.2, zorder=3
    )
    ax.add_patch(b)
    ax.text(13.5, cur_y - h/2, text, color=border, fontsize=8.2, fontweight='bold', ha='center', va='center', zorder=4, linespacing=1.25)
    cur_y -= (h + 0.16)

# ลูกศรเชื่อมจาก Train Box -> Model
ax.annotate('', xy=(11.75, 6.6), xytext=(11.25, 6.6),
            arrowprops=dict(arrowstyle="-|>", color="#1D4ED8", lw=2.2, mutation_scale=15), zorder=5)
# ลูกศรเชื่อมจาก Val Box -> Model
ax.annotate('', xy=(11.75, 4.7), xytext=(11.25, 4.7),
            arrowprops=dict(arrowstyle="-|>", color="#B45309", lw=2.2, mutation_scale=15), zorder=5)

# ==========================================
# 6. Mini Table: รายละเอียด 8 คลาส (ซ้ายล่าง)
# ==========================================
table_bg = patches.FancyBboxPatch(
    (0.8, 1.6), 6.0, 4.3,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5, zorder=2
)
ax.add_patch(table_bg)

tbl_hdr = patches.FancyBboxPatch(
    (0.8, 5.45), 6.0, 0.45,
    boxstyle="round,pad=0.08,rounding_size=0.15",
    facecolor="#1E293B", edgecolor="none", zorder=3
)
ax.add_patch(tbl_hdr)
ax.text(3.8, 5.67, "ตารางแจกแจงจำนวนภาพรายโรคทั้ง 8 คลาส (Stratified)", color="#FFFFFF", fontsize=11, fontweight='bold', ha='center', va='center', zorder=4)

# หัวตาราง
ty = 5.2
ax.text(1.0, ty, "ชื่อโรค (Class Name)", fontsize=8.5, fontweight='bold', color="#475569", va='center')
ax.text(3.5, ty, "Train (70%)", fontsize=8.5, fontweight='bold', color="#1D4ED8", ha='center', va='center')
ax.text(4.45, ty, "Val (20%)", fontsize=8.5, fontweight='bold', color="#B45309", ha='center', va='center')
ax.text(5.35, ty, "Test (10%)", fontsize=8.5, fontweight='bold', color="#047857", ha='center', va='center')
ax.text(6.25, ty, "รวมทั้งหมด", fontsize=8.5, fontweight='bold', color="#0F172A", ha='center', va='center')

line = patches.Rectangle((0.9, ty - 0.12), 5.8, 0.015, facecolor="#CBD5E1", zorder=3)
ax.add_patch(line)

classes_data = [
    ("1. Melanoma (มะเร็งเมลาโนมา)", "5,728", "1,636", "819", "8,183"),
    ("2. Melanocytic Nevi (ไฝ/ปาน)", "5,719", "1,634", "817", "8,170"),
    ("3. Keratosis (กระเนื้อ)", "2,900", "828", "415", "4,143"),
    ("4. BCC (มะเร็งเบซัลเซลล์)", "2,326", "664", "333", "3,323"),
    ("5. SCC (มะเร็งสความัสเซลล์)", "2,121", "606", "304", "3,031"),
    ("6. Actinic Keratosis (ก่อนมะเร็ง)", "1,965", "561", "282", "2,808"),
    ("7. Psoriasis (สะเก็ดเงิน)", "1,438", "411", "206", "2,055"),
    ("8. Tinea (เชื้อราผิวหนัง)", "1,334", "381", "191", "1,906"),
]

row_y = 4.85
for cname, tr, va, te, tot in classes_data:
    ax.text(1.0, row_y, cname, fontsize=8, color="#1E293B", va='center')
    ax.text(3.5, row_y, tr, fontsize=8, color="#1D4ED8", ha='center', va='center')
    ax.text(4.45, row_y, va, fontsize=8, color="#B45309", ha='center', va='center')
    ax.text(5.35, row_y, te, fontsize=8, color="#047857", ha='center', va='center')
    ax.text(6.25, row_y, tot, fontsize=8, fontweight='bold', color="#0F172A", ha='center', va='center')
    row_y -= 0.34

# ขีดเส้นสรุปรวม
sum_line = patches.Rectangle((0.9, row_y + 0.12), 5.8, 0.015, facecolor="#94A3B8", zorder=3)
ax.add_patch(sum_line)

ax.text(1.0, row_y - 0.05, "รวมทั้งหมด (Total)", fontsize=8.5, fontweight='bold', color="#0F172A", va='center')
ax.text(3.5, row_y - 0.05, "23,531", fontsize=8.5, fontweight='bold', color="#1D4ED8", ha='center', va='center')
ax.text(4.45, row_y - 0.05, "6,721", fontsize=8.5, fontweight='bold', color="#B45309", ha='center', va='center')
ax.text(5.35, row_y - 0.05, "3,367", fontsize=8.5, fontweight='bold', color="#047857", ha='center', va='center')
ax.text(6.25, row_y - 0.05, "33,619", fontsize=9, fontweight='bold', color="#4338CA", ha='center', va='center')

# Footer
ax.text(8.0, 0.7, "ปริญญานิพนธ์: โมบายแอปพลิเคชันประเมินความเสี่ยงมะเร็งผิวหนังและรอยโรคทั่วไป | สาขาวิศวกรรมคอมพิวเตอร์ มทร.อีสาน",
        color="#64748B", fontsize=9.5, ha='center', va='center')

# บันทึกภาพ
output_path = "dataset_split_resnet50.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated: {output_path}")
