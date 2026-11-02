import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.family'] = 'Leelawadee UI'
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# รูปที่ 1: ทฤษฎี Class Weights (การแก้ปัญหาข้อมูลไม่สมดุล)
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(13, 7.5), dpi=300, facecolor='#F8FAFC')
ax1.set_xlim(0, 13)
ax1.set_ylim(0, 7.5)
ax1.axis('off')

# Header
hdr1 = patches.FancyBboxPatch((0.6, 6.6), 11.8, 0.65, boxstyle="round,pad=0.06,rounding_size=0.12",
                             facecolor="#0F172A", edgecolor="none")
ax1.add_patch(hdr1)
ax1.text(6.5, 6.95, "ทฤษฎีการถ่วงน้ำหนักคลาส (Balanced Class Weights) แก้ปัญหา Class Imbalance",
         color="#FFFFFF", fontsize=14, fontweight='bold', ha='center', va='center')

# ฝั่งซ้าย: ปัญหาข้อมูลไม่สมดุล (Data Imbalance)
left_card = patches.FancyBboxPatch((0.6, 1.2), 5.6, 5.1, boxstyle="round,pad=0.08,rounding_size=0.15",
                                   facecolor="#FFFFFF", edgecolor="#EF4444", linewidth=1.8)
ax1.add_patch(left_card)
ax1.text(3.4, 5.95, "1. สภาพปัญหา: ข้อมูลไม่สมดุล (Class Imbalance)", color="#DC2626", fontsize=12, fontweight='bold', ha='center')

# แท่งเปรียบเทียบข้อมูล
ax1.text(1.0, 5.4, "คลาสส่วนใหญ่ (Majority Class): Melanoma (~8,183 ภาพ)", fontsize=9.5, fontweight='bold', color="#1E293B")
b1 = patches.Rectangle((1.0, 4.95), 4.8, 0.35, facecolor="#3B82F6", edgecolor="#1D4ED8")
ax1.add_patch(b1)
ax1.text(3.4, 5.12, "มีข้อมูลเยอะมาก (48% ของ Dataset)", color="#FFFFFF", fontsize=9, fontweight='bold', ha='center', va='center')

ax1.text(1.0, 4.5, "คลาสส่วนน้อย (Minority Class): Tinea / Psoriasis (~1,906 ภาพ)", fontsize=9.5, fontweight='bold', color="#1E293B")
b2 = patches.Rectangle((1.0, 4.05), 1.3, 0.35, facecolor="#F97316", edgecolor="#C2410C")
ax1.add_patch(b2)
ax1.text(1.65, 4.22, "ข้อมูลน้อย", color="#FFFFFF", fontsize=8.5, fontweight='bold', ha='center', va='center')

# กล่องผลกระทบหากไม่ใช้
err_box = patches.FancyBboxPatch((1.0, 1.45), 4.8, 2.15, boxstyle="round,pad=0.06,rounding_size=0.1",
                                 facecolor="#FEF2F2", edgecolor="#FCA5A5", linewidth=1.2)
ax1.add_patch(err_box)
ax1.text(3.4, 3.32, "ผลเสียหากฝึกสอนแบบปกติ (No Class Weights):", color="#991B1B", fontsize=10, fontweight='bold', ha='center')
ax1.text(1.2, 2.30, "• โมเดลถูกครอบงำด้วยคลาสส่วนใหญ่ (Model Bias)\n• โมเดลเลือกทายแต่คลาสที่มีภาพเยอะ เพื่อให้ได้ Accuracy สูง\n• ตรวจจับโรคส่วนน้อย (เช่น เชื้อรา/สะเก็ดเงิน) ได้ต่ำมาก\n• เกิดสภาวะ 'ความแม่นยำลวงตา' (Misleading Accuracy)",
         color="#7F1D1D", fontsize=8.8, linespacing=1.35, va='center')

# ฝั่งขวา: การแก้ปัญหาด้วย Class Weights
right_card = patches.FancyBboxPatch((6.8, 1.2), 5.6, 5.1, boxstyle="round,pad=0.08,rounding_size=0.15",
                                    facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.8)
ax1.add_patch(right_card)
ax1.text(9.6, 5.95, "2. การแก้ไข: ถ่วงน้ำหนักใน Loss Function", color="#059669", fontsize=12, fontweight='bold', ha='center')

# กล่องสูตรคำนวณ
eq_box = patches.FancyBboxPatch((7.2, 4.65), 4.8, 1.05, boxstyle="round,pad=0.06,rounding_size=0.1",
                                facecolor="#ECFDF5", edgecolor="#6EE7B7", linewidth=1.2)
ax1.add_patch(eq_box)
ax1.text(9.6, 5.37, "สูตรคำนวณค่าน้ำหนักชดเชย (สมการที่ 2.4 ในเล่ม):", color="#065F46", fontsize=9.5, fontweight='bold', ha='center')
ax1.text(9.6, 4.95, r"$W_j = \left( \frac{1}{N_j} \right) \times \left( \frac{N_{total}}{K} \right)$",
         color="#047857", fontsize=14, fontweight='bold', ha='center', va='center')

# การเปรียบเทียบค่าน้ำหนัก
ax1.text(7.2, 4.25, "• คลาสภาพน้อย (Tinea): จะได้ Weight สูง (โทษหนักเมื่อทายผิด)", fontsize=9, color="#1E293B")
ax1.text(7.2, 3.85, "• คลาสภาพมาก (Melanoma): จะได้ Weight ต่ำ (ปรับลดบทลงโทษ)", fontsize=9, color="#1E293B")

# กล่องผลลัพธ์ที่ดี
res_box = patches.FancyBboxPatch((7.2, 1.45), 4.8, 2.15, boxstyle="round,pad=0.06,rounding_size=0.1",
                                 facecolor="#EFF6FF", edgecolor="#93C5FD", linewidth=1.2)
ax1.add_patch(res_box)
ax1.text(9.6, 3.32, "ประโยชน์จากการใส่ Balanced Class Weights:", color="#1E40AF", fontsize=10, fontweight='bold', ha='center')
ax1.text(7.4, 2.30, "1. เพิ่มอัตราการลงโทษ (Heavy Penalty) เมื่อทายคลาสส่วนน้อยผิด\n2. บังคับให้เกรเดียนต์ปรับปรุงน้ำหนักเพื่อเรียนรู้ทุกโรคเท่าเทียมกัน\n3. ค่า Macro F1-score สูงถึง 76.99% กระจายความแม่นยำทุกโรค\n4. ป้องกันโมเดลเอนเอียง (Bias) ไปยังโรคใดโรคหนึ่ง",
         color="#1E3A8A", fontsize=8.8, linespacing=1.35, va='center')

# ลูกศรตรงกลาง
ax1.annotate('', xy=(6.7, 3.75), xytext=(6.3, 3.75),
             arrowprops=dict(arrowstyle="-|>", color="#059669", lw=3, mutation_scale=18))

# Footer
ax1.text(6.5, 0.7, "โมบายแอปพลิเคชันประเมินความเสี่ยงมะเร็งผิวหนังและรอยโรคทั่วไป | วิศวกรรมคอมพิวเตอร์ มทร.อีสาน",
         color="#64748B", fontsize=9.5, ha='center')

fig1_path = "theory_class_weights.png"
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"Generated: {fig1_path}")


# ==============================================================================
# รูปที่ 2: ทฤษฎี Attention Block (Channel Attention / SE-Block)
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(14, 7.5), dpi=300, facecolor='#F8FAFC')
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 7.5)
ax2.axis('off')

hdr2 = patches.FancyBboxPatch((0.6, 6.6), 12.8, 0.65, boxstyle="round,pad=0.06,rounding_size=0.12",
                             facecolor="#0F172A", edgecolor="none")
ax2.add_patch(hdr2)
ax2.text(7.0, 6.95, "โครงสร้างกลไกความใส่ใจระดับช่องสัญญาณ (Squeeze-and-Excitation Attention Block)",
         color="#FFFFFF", fontsize=14, fontweight='bold', ha='center', va='center')

# 1. Input Feature Map
in_box = patches.FancyBboxPatch((0.6, 2.4), 2.2, 3.5, boxstyle="round,pad=0.06,rounding_size=0.15",
                                facecolor="#FFFFFF", edgecolor="#3B82F6", linewidth=1.8)
ax2.add_patch(in_box)
ax2.text(1.7, 5.5, "Input Feature Map", color="#1D4ED8", fontsize=11, fontweight='bold', ha='center')
ax2.text(1.7, 5.15, "(จาก ResNet-50)", color="#64748B", fontsize=9, ha='center')

fmap_mock = patches.Rectangle((1.0, 3.2), 1.4, 1.6, facecolor="#DBEAFE", edgecolor="#3B82F6", lw=1.2)
ax2.add_patch(fmap_mock)
ax2.text(1.7, 4.0, "X\n[H × W × C]", color="#1E40AF", fontsize=11, fontweight='bold', ha='center', va='center')
ax2.text(1.7, 2.75, "มีฟีเจอร์รอยโรค\n+ สิ่งรบกวน (ขน/เงา)", color="#475569", fontsize=8.5, ha='center')

# ลูกศร 1 -> 2
ax2.annotate('', xy=(3.3, 4.15), xytext=(2.9, 4.15),
             arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=2, mutation_scale=15))

# 2. Squeeze Operation
sq_box = patches.FancyBboxPatch((3.4, 2.4), 2.8, 3.5, boxstyle="round,pad=0.06,rounding_size=0.15",
                                facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.5)
ax2.add_patch(sq_box)
ax2.text(4.8, 5.5, "1. Squeeze (บีบอัดมิติ)", color="#1E3A8A", fontsize=11, fontweight='bold', ha='center')
ax2.text(4.8, 5.15, "GlobalAveragePooling2D", color="#2563EB", fontsize=9, ha='center')

sq_rect = patches.Rectangle((4.2, 3.5), 1.2, 1.2, facecolor="#BFDBFE", edgecolor="#2563EB", lw=1.2)
ax2.add_patch(sq_rect)
ax2.text(4.8, 4.1, "1 × 1 × C\n(Vector z)", color="#1E3A8A", fontsize=10, fontweight='bold', ha='center', va='center')
ax2.text(4.8, 2.85, "ยุบข้อมูลพื้นที่ (H × W)\nให้เหลือตัวแทนสถิติเฉลี่ย\n1 ค่าต่อ 1 Channel", color="#334155", fontsize=8.5, ha='center', linespacing=1.2)

# ลูกศร 2 -> 3
ax2.annotate('', xy=(6.7, 4.15), xytext=(6.3, 4.15),
             arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=2, mutation_scale=15))

# 3. Excitation Operation
ex_box = patches.FancyBboxPatch((6.8, 2.4), 3.4, 3.5, boxstyle="round,pad=0.06,rounding_size=0.15",
                                facecolor="#FFFBEB", edgecolor="#D97706", linewidth=1.5)
ax2.add_patch(ex_box)
ax2.text(8.5, 5.5, "2. Excitation (คำนวณความใส่ใจ)", color="#92400E", fontsize=11, fontweight='bold', ha='center')
ax2.text(8.5, 5.15, "Dense + ReLU + Dense + Sigmoid", color="#B45309", fontsize=8.5, ha='center')

sub1 = patches.Rectangle((7.1, 4.1), 2.8, 0.75, facecolor="#FEF3C7", edgecolor="#D97706", lw=1)
ax2.add_patch(sub1)
ax2.text(8.5, 4.48, "Dense(C/16) + ReLU (ลดมิติ)", color="#92400E", fontsize=8.5, fontweight='bold', ha='center', va='center')

sub2 = patches.Rectangle((7.1, 3.15), 2.8, 0.75, facecolor="#FDE68A", edgecolor="#D97706", lw=1)
ax2.add_patch(sub2)
ax2.text(8.5, 3.52, "Dense(C) + Sigmoid (คืนขนาดเดิม)", color="#78350F", fontsize=8.5, fontweight='bold', ha='center', va='center')

ax2.text(8.5, 2.7, "สร้างค่าน้ำหนักความสนใจ s\n(ช่วงคะแนน 0 ถึง 1 ในแต่ละแชนแนล)", color="#451A03", fontsize=8.5, ha='center')

# ลูกศร 3 -> 4
ax2.annotate('', xy=(10.7, 4.15), xytext=(10.3, 4.15),
             arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=2, mutation_scale=15))

# 4. Scale (Re-weighting)
sc_box = patches.FancyBboxPatch((10.8, 2.4), 2.6, 3.5, boxstyle="round,pad=0.06,rounding_size=0.15",
                                facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.8)
ax2.add_patch(sc_box)
ax2.text(12.1, 5.5, "3. Scale (ปรับสเกล)", color="#065F46", fontsize=11, fontweight='bold', ha='center')
ax2.text(12.1, 5.15, "Element-wise Multiply", color="#059669", fontsize=8.5, ha='center')

sc_rect = patches.Rectangle((11.4, 3.3), 1.4, 1.4, facecolor="#A7F3D0", edgecolor="#059669", lw=1.2)
ax2.add_patch(sc_rect)
ax2.text(12.1, 4.0, r"$\tilde{X} = X \cdot s$", color="#064E3B", fontsize=13, fontweight='bold', ha='center', va='center')

ax2.text(12.1, 2.85, "• เน้นย้ำฟีเจอร์รอยโรค\n• ลดทอนขน/เงา/ผิวปกติ", color="#047857", fontsize=9, fontweight='bold', ha='center', linespacing=1.2)

# กล่องผลลัพธ์ล่างสุด
bot_box = patches.FancyBboxPatch((0.6, 1.05), 12.8, 1.05, boxstyle="round,pad=0.06,rounding_size=0.1",
                                 facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.2)
ax2.add_patch(bot_box)
ax2.text(7.0, 1.7, "บทบาทสำคัญของ SE-Attention Block ในโปรเจกต์นี้:", color="#0F172A", fontsize=10.5, fontweight='bold', ha='center')
ax2.text(7.0, 1.3, "รอยโรคผิวหนังมักมีสิ่งรบกวนตามธรรมชาติ (เช่น ขน แสงสะท้อน ผิวหนังปกติรอบข้าง) กลไกนี้ช่วยคัดกรองสัญญาณรบกวนออก\nทำให้โมเดล ResNet-50 จดจำเฉพาะ 'ขอบเขตและเม็ดสีจริงของรอยโรค' ช่วยเพิ่ม Accuracy ขึ้นจาก 76.60% เป็น 81.05% (+4.45%)",
         color="#334155", fontsize=9, ha='center', linespacing=1.3)

fig2_path = "theory_attention_block.png"
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"Generated: {fig2_path}")


# ==============================================================================
# รูปที่ 3: ทฤษฎี Two-Phase Training (Feature Extraction & Fine-Tuning)
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=(14, 7.8), dpi=300, facecolor='#F8FAFC')
ax3.set_xlim(0, 14)
ax3.set_ylim(0, 7.8)
ax3.axis('off')

hdr3 = patches.FancyBboxPatch((0.6, 6.9), 12.8, 0.65, boxstyle="round,pad=0.06,rounding_size=0.12",
                             facecolor="#0F172A", edgecolor="none")
ax3.add_patch(hdr3)
ax3.text(7.0, 7.23, "กลยุทธ์การฝึกสอนแบบสองเฟส (Two-Phase Transfer Learning Strategy)",
         color="#FFFFFF", fontsize=14, fontweight='bold', ha='center', va='center')

# การ์ดเฟส 1 (ซ้าย)
card_p1 = patches.FancyBboxPatch((0.6, 1.8), 6.1, 4.8, boxstyle="round,pad=0.08,rounding_size=0.15",
                                 facecolor="#FFFFFF", edgecolor="#2563EB", linewidth=1.8)
ax3.add_patch(card_p1)
ax3.text(3.65, 6.25, "ระยะที่ 1 : Feature Extraction", color="#1D4ED8", fontsize=13, fontweight='bold', ha='center')
ax3.text(3.65, 5.9, "ฝึกสอนเฉพาะส่วนหัวใหม่ (วอร์มอัป Classifier Head)", color="#475569", fontsize=9.5, ha='center')

# เลเยอร์ในเฟส 1
l1_freeze = patches.FancyBboxPatch((1.0, 4.0), 5.3, 1.6, boxstyle="round,pad=0.06,rounding_size=0.1",
                                   facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.5, linestyle="--")
ax3.add_patch(l1_freeze)
ax3.text(3.65, 5.15, "[แช่แข็ง] ResNet-50 Backbone ทั้งหมด (50 เลเยอร์)", color="#1E40AF", fontsize=10.5, fontweight='bold', ha='center')
ax3.text(3.65, 4.75, "แช่แข็งค่าน้ำหนัก 100% (base_model.trainable = False)", color="#DC2626", fontsize=9.5, fontweight='bold', ha='center')
ax3.text(3.65, 4.35, "รักษาความรู้เดิมจาก ImageNet (ป้องกัน Catastrophic Forgetting)", color="#475569", fontsize=8.5, ha='center')

l1_head = patches.FancyBboxPatch((1.0, 2.7), 5.3, 1.1, boxstyle="round,pad=0.06,rounding_size=0.1",
                                 facecolor="#DCFCE7", edgecolor="#16A34A", linewidth=1.5)
ax3.add_patch(l1_head)
ax3.text(3.65, 3.45, "[เปิดเทรน] ส่วนหัวที่ต่อเพิ่มใหม่ (Head Classifier)", color="#15803D", fontsize=10, fontweight='bold', ha='center')
ax3.text(3.65, 3.1, "SE-Attention + GlobalAvgPool + Dense 256 + Softmax (8 คลาส)", color="#166534", fontsize=8.5, ha='center')
ax3.text(3.65, 2.82, "สถานะ: เปิดให้เทรนเพื่อปรับตัวเข้ากับข้อมูลรอยโรค", color="#14532D", fontsize=8, ha='center')

ax3.text(3.65, 2.3, "• อัตราการเรียนรู้ (Learning Rate): 0.001 (1e-3)\n• Optimizer: Adam | Loss: Categorical Crossentropy",
         color="#1E293B", fontsize=9, ha='center', linespacing=1.25)


# การ์ดเฟส 2 (ขวา)
card_p2 = patches.FancyBboxPatch((7.3, 1.8), 6.1, 4.8, boxstyle="round,pad=0.08,rounding_size=0.15",
                                 facecolor="#FFFFFF", edgecolor="#EA580C", linewidth=1.8)
ax3.add_patch(card_p2)
ax3.text(10.35, 6.25, "ระยะที่ 2 : Fine-Tuning", color="#C2410C", fontsize=13, fontweight='bold', ha='center')
ax3.text(10.35, 5.9, "ปรับจูนเลเยอร์ระดับสูงอย่างละเอียด (Fine-tune High-level)", color="#475569", fontsize=9.5, ha='center')

# เลเยอร์ในเฟส 2
l2_freeze = patches.FancyBboxPatch((7.7, 4.8), 5.3, 0.85, boxstyle="round,pad=0.06,rounding_size=0.08",
                                   facecolor="#EFF6FF", edgecolor="#3B82F6", linewidth=1.2, linestyle="--")
ax3.add_patch(l2_freeze)
ax3.text(10.35, 5.35, "[แช่แข็ง] 120 เลเยอร์แรกของ ResNet-50", color="#1E40AF", fontsize=9.5, fontweight='bold', ha='center')
ax3.text(10.35, 5.05, "ล็อกไว้คงเดิม (สกัด Low-level Features: เส้น ขอบ สี พื้นฐาน)", color="#475569", fontsize=8.5, ha='center')

l2_unfreeze = patches.FancyBboxPatch((7.7, 3.75), 5.3, 0.9, boxstyle="round,pad=0.06,rounding_size=0.08",
                                     facecolor="#FFEDD5", edgecolor="#EA580C", linewidth=1.5)
ax3.add_patch(l2_unfreeze)
ax3.text(10.35, 4.35, "[ปลดล็อกเทรน] เลเยอร์ระดับสูง (> เลเยอร์ที่ 120 ขึ้นไป)", color="#C2410C", fontsize=9.5, fontweight='bold', ha='center')
ax3.text(10.35, 4.0, "ปลดล็อกให้ร่วมเรียนรู้ลายรอยโรคผิวหนังเฉพาะทาง", color="#9A3412", fontsize=8.5, ha='center')

l2_head = patches.FancyBboxPatch((7.7, 2.7), 5.3, 0.9, boxstyle="round,pad=0.06,rounding_size=0.08",
                                 facecolor="#DCFCE7", edgecolor="#16A34A", linewidth=1.2)
ax3.add_patch(l2_head)
ax3.text(10.35, 3.3, "[เปิดเทรน] ส่วนหัวที่ต่อเพิ่มใหม่ (Head Classifier)", color="#15803D", fontsize=9.5, fontweight='bold', ha='center')
ax3.text(10.35, 2.95, "เรียนรู้ต่อยอดร่วมกับเลเยอร์ส่วนท้ายอย่างลงตัว", color="#166534", fontsize=8.5, ha='center')

ax3.text(10.35, 2.3, "• อัตราการเรียนรู้ (Learning Rate): 0.00001 (1e-5) [ลดลง 100 เท่า!]\n• วัตถุประสงค์: ค่อยๆ ปรับจูนค่าน้ำหนัก ไม่ให้กระโดดข้ามจุดเหมาะสม",
         color="#1E293B", fontsize=9, ha='center', linespacing=1.25)

# ลูกศรเชื่อมระหว่างเฟส 1 -> เฟส 2
ax3.annotate('', xy=(7.2, 4.2), xytext=(6.8, 4.2),
             arrowprops=dict(arrowstyle="-|>", color="#EA580C", lw=3, mutation_scale=18))

# การ์ดสรุปผลด้านล่าง
summary_box = patches.FancyBboxPatch((0.6, 0.5), 12.8, 1.05, boxstyle="round,pad=0.06,rounding_size=0.1",
                                     facecolor="#F1F5F9", edgecolor="#64748B", linewidth=1.2)
ax3.add_patch(summary_box)
ax3.text(7.0, 1.18, "ผลลัพธ์จากการใช้ Two-Phase Training ในโครงงานนี้ (ดูรูป 4.1 และ 4.2 ในเล่มหน้า 98–99):",
         color="#0F172A", fontsize=10, fontweight='bold', ha='center')
ax3.text(7.0, 0.78, "• หลังจบ Phase 1 (รอบที่ 1–39): Validation Accuracy ไต่ระดับจาก 40% ไปถึง 76.50%\n• เมื่อเข้าสู่ Phase 2 Fine-Tuning (รอบที่ 39–60): ปลดล็อกเลเยอร์ท้ายร่วมกับ LR=1e-5 ทำให้ Accuracy พุ่งขึ้นสู่ 81.05% อย่างมั่นคง ไม่เกิด Overfitting",
         color="#334155", fontsize=9, ha='center', linespacing=1.3)

fig3_path = "theory_two_phase_training.png"
fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close(fig3)
print(f"Generated: {fig3_path}")
