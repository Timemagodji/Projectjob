# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ตั้งค่าฟอนต์ภาษาไทย
plt.rcParams['font.sans-serif'] = ['Leelawadee UI', 'Tahoma', 'Segoe UI', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 9), dpi=300, facecolor='#F8FAFC')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Header
hdr = patches.FancyBboxPatch((0.6, 8.05), 14.8, 0.7, boxstyle="round,pad=0.06,rounding_size=0.12",
                             facecolor="#0F172A", edgecolor="none")
ax.add_patch(hdr)
ax.text(8.0, 8.4, "โครงสร้าง 50 เลเยอร์ของ ResNet-50 (ที่มาของชื่อ ResNet-50 และหน้าที่ของแต่ละชั้น)",
        color="#FFFFFF", fontsize=14.5, fontweight='bold', ha='center', va='center')

# ==============================================================================
# ด้านบน: แผนภาพโฟลว์แนวนอน (Horizontal Pipeline Flow)
# ==============================================================================
flow_card = patches.FancyBboxPatch((0.6, 5.0), 14.8, 2.85, boxstyle="round,pad=0.06,rounding_size=0.12",
                                  facecolor="#FFFFFF", edgecolor="#3B82F6", linewidth=1.5)
ax.add_patch(flow_card)
ax.text(1.0, 7.55, "แผนผังการไหลของข้อมูลและมิติภาพในแต่ละช่วง (Pipeline Flow):", color="#1E40AF", fontsize=11.5, fontweight='bold')

pipe_stages = [
    ("ภาพนำเข้า\n(Input)", "224 × 224\n3 ช่องสี (RGB)", "#F1F5F9", "#475569", 1.2),
    ("Stage 1\nConv1 + Pool", "1 ชั้น (Conv)\nลดเหลือ 56×56", "#DBEAFE", "#1E40AF", 3.2),
    ("Stage 2\nConv2_x", "3 บล็อก\n(รวม 9 ชั้น)", "#E0E7FF", "#3730A3", 5.4),
    ("Stage 3\nConv3_x", "4 บล็อก\n(รวม 12 ชั้น)", "#EDE9FE", "#5B21B6", 7.6),
    ("Stage 4\nConv4_x", "6 บล็อก\n(รวม 18 ชั้น)", "#FCE7F3", "#9D174D", 9.8),
    ("Stage 5\nConv5_x", "3 บล็อก\n(รวม 9 ชั้น)", "#FEE2E2", "#991B1B", 12.0),
    ("ส่วนหัวใหม่\nClassifier", "1 ชั้น (Dense)\nทาย 8 โรค", "#DCFCE7", "#166534", 14.2)
]

for title, sub, bg, fg, x in pipe_stages:
    box = patches.FancyBboxPatch((x - 0.75, 5.3), 1.5, 1.8, boxstyle="round,pad=0.06,rounding_size=0.1",
                                facecolor=bg, edgecolor=fg, lw=1.5)
    ax.add_patch(box)
    ax.text(x, 6.55, title, color=fg, fontsize=9.5, fontweight='bold', ha='center', va='center')
    ax.plot([x - 0.55, x + 0.55], [6.05, 6.05], color=fg, lw=0.8, alpha=0.5)
    ax.text(x, 5.65, sub, color="#334155", fontsize=8.2, ha='center', va='center')
    
    if x < 14.0:
        ax.annotate('', xy=(x + 1.25, 6.2), xytext=(x + 0.85, 6.2),
                    arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=2, mutation_scale=12))

# ==============================================================================
# ด้านล่าง: ตารางแจกแจงการนับเลเยอร์ให้ครบ 50 ชั้น (Layer-by-Layer Breakdown)
# ==============================================================================
table_card = patches.FancyBboxPatch((0.6, 0.8), 14.8, 4.0, boxstyle="round,pad=0.06,rounding_size=0.12",
                                   facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.5)
ax.add_patch(table_card)
ax.text(1.0, 4.45, "ตารางแจกแจงจำนวนเลเยอร์จริง ทำไมถึงนับได้ 50 เลเยอร์? (1 + 9 + 12 + 18 + 9 + 1 = 50):",
        color="#065F46", fontsize=11.5, fontweight='bold')

col_x = [0.9, 3.1, 4.8, 8.0, 9.5, 12.3]

# หัวตาราง
tbl_hdr = patches.Rectangle((0.8, 4.0), 14.4, 0.35, facecolor="#065F46", edgecolor="none")
ax.add_patch(tbl_hdr)
tbl_titles = ["ช่วงของโมเดล", "มิติของภาพ", "โครงสร้างบล็อก", "จำนวนบล็อก", "จำนวนเลเยอร์ (ชั้น)", "หน้าที่ในการสกัดฟีเจอร์ภาพ"]
for cx, t in zip(col_x, tbl_titles):
    ax.text(cx + 0.05, 4.18, t, color="#FFFFFF", fontsize=8.8, fontweight='bold', va='center')

rows = [
    ("Stage 1 (Conv1)", "112 × 112", "7×7 Convolution (Stride 2)", "-", "1 เลเยอร์", "ตรวจจับเส้นตรง ขอบภาพ และแสงเงาพื้นฐาน", "#F8FAFC"),
    ("Stage 2 (Conv2_x)", "56 × 56", "1×1 (64) -> 3×3 (64) -> 1×1 (256)", "3 บล็อก", "3 บล็อก × 3 ชั้น = 9 เลเยอร์", "รวมเส้นเป็นมุม ลวดลาย และพื้นผิวทั่วไป", "#F0FDF4"),
    ("Stage 3 (Conv3_x)", "28 × 28", "1×1 (128) -> 3×3 (128) -> 1×1 (512)", "4 บล็อก", "4 บล็อก × 3 ชั้น = 12 เลเยอร์", "สกัดรูปทรงเรขาคณิต ส่วนโค้ง ความนูน", "#F8FAFC"),
    ("Stage 4 (Conv4_x)", "14 × 14", "1×1 (256) -> 3×3 (256) -> 1×1 (1024)", "6 บล็อก", "6 บล็อก × 3 ชั้น = 18 เลเยอร์", "สกัดลักษณะขอบรอยโรค รูปร่างแผลผิวหนัง", "#FEF3C7"),
    ("Stage 5 (Conv5_x)", "7 × 7", "1×1 (512) -> 3×3 (512) -> 1×1 (2048)", "3 บล็อก", "3 บล็อก × 3 ชั้น = 9 เลเยอร์", "สกัดแพทเทิร์นเม็ดสีลึก และโครงสร้างจำเพาะโรค", "#FEE2E2"),
    ("Classifier Head", "1 × 1", "Dense Output (Softmax 8 Classes)", "-", "1 เลเยอร์", "คำนวณความน่าจะเป็นของโรคผิวหนังทั้ง 8 โรค", "#DCFCE7")
]

ry = 3.65
for r in rows:
    bg_rect = patches.Rectangle((0.8, ry - 0.1), 14.4, 0.42, facecolor=r[6], edgecolor="#E2E8F0", lw=0.5)
    ax.add_patch(bg_rect)
    ax.text(col_x[0] + 0.1, ry + 0.1, r[0], fontsize=8.5, fontweight='bold', color="#1E293B", va='center')
    ax.text(col_x[1] + 0.1, ry + 0.1, r[1], fontsize=8.2, color="#334155", va='center')
    ax.text(col_x[2] + 0.1, ry + 0.1, r[2], fontsize=8.2, color="#0F172A", va='center')
    ax.text(col_x[3] + 0.3, ry + 0.1, r[3], fontsize=8.2, color="#475569", va='center')
    ax.text(col_x[4] + 0.1, ry + 0.1, r[4], fontsize=8.5, fontweight='bold', color="#047857" if "เลเยอร์" in r[4] else "#1E293B", va='center')
    ax.text(col_x[5] + 0.1, ry + 0.1, r[5], fontsize=8.2, color="#334155", va='center')
    ry -= 0.42

# สรุปผลรวมด้านล่างตาราง
sum_box = patches.Rectangle((0.8, 1.0), 14.4, 0.45, facecolor="#047857", edgecolor="none")
ax.add_patch(sum_box)
ax.text(8.0, 1.23, "ผลรวมเลเยอร์ที่มีค่าน้ำหนักทั้งหมด: 1 + 9 + 12 + 18 + 9 + 1 = 50 เลเยอร์ (สถาปัตยกรรม ResNet-50)",
        color="#FFFFFF", fontsize=10.5, fontweight='bold', ha='center', va='center')

# Footer
ax.text(8.0, 0.35, "โมบายแอปพลิเคชันประเมินความเสี่ยงมะเร็งผิวหนังและรอยโรคทั่วไป | วิศวกรรมคอมพิวเตอร์ มทร.อีสาน",
        color="#64748B", fontsize=9, ha='center')

out_file = "resnet50_layers_breakdown.png"
fig.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Generated: {out_file}")
