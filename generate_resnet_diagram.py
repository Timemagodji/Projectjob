import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ตั้งค่าฟอนต์ภาษาไทย
plt.rcParams['font.sans-serif'] = ['Leelawadee UI', 'Segoe UI', 'DejaVu Sans', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300, facecolor='#F8FAFC')
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis('off')

# Header
hdr = patches.FancyBboxPatch((0.6, 7.6), 13.8, 0.65, boxstyle="round,pad=0.06,rounding_size=0.12",
                             facecolor="#0F172A", edgecolor="none")
ax.add_patch(hdr)
ax.text(7.5, 7.95, "สถาปัตยกรรมโครงข่าย ResNet-50 (Deep Residual Network)",
        color="#FFFFFF", fontsize=14, fontweight='bold', ha='center', va='center')

# ==============================================================================
# ฝั่งซ้าย: โครงสร้างภาพรวมของ ResNet-50 (Overall Macro Architecture)
# ==============================================================================
card_left = patches.FancyBboxPatch((0.6, 1.0), 6.8, 6.3, boxstyle="round,pad=0.08,rounding_size=0.15",
                                  facecolor="#FFFFFF", edgecolor="#3B82F6", linewidth=1.8)
ax.add_patch(card_left)
ax.text(4.0, 7.0, "1. โครงสร้างรวม 50 เลเยอร์ (Macro Architecture)", color="#1D4ED8", fontsize=12, fontweight='bold', ha='center')

stages = [
    ("Input Image", "ขนาด 224 × 224 × 3", "#F1F5F9", "#475569"),
    ("Stage 1: Conv1 + MaxPool", "7×7 Conv (64), Stride 2 (เหลือ 112×112) | MaxPool (เหลือ 56×56)", "#DBEAFE", "#1E40AF"),
    ("Stage 2: Conv2_x (3 Blocks)", "3 × [1×1(64), 3×3(64), 1×1(256)] (ขนาด 56×56, รวม 9 เลเยอร์)", "#E0E7FF", "#3730A3"),
    ("Stage 3: Conv3_x (4 Blocks)", "4 × [1×1(128), 3×3(128), 1×1(512)] (ขนาด 28×28, รวม 12 เลเยอร์)", "#EDE9FE", "#5B21B6"),
    ("Stage 4: Conv4_x (6 Blocks)", "6 × [1×1(256), 3×3(256), 1×1(1024)] (ขนาด 14×14, รวม 18 เลเยอร์) *ปลดล็อกเทรน Phase 2*", "#FCE7F3", "#9D174D"),
    ("Stage 5: Conv5_x (3 Blocks)", "3 × [1×1(512), 3×3(512), 1×1(2048)] (ขนาด 7×7, รวม 9 เลเยอร์) *ปลดล็อกเทรน Phase 2*", "#FEE2E2", "#991B1B"),
    ("Custom Head 8 คลาส (ต่อเติมใหม่)", "SE-Attention + GAP + Dense 256 + Dense 8 (Softmax) *เทรนทั้งสองเฟส*", "#DCFCE7", "#166534")
]

y_pos = 6.25
for name, desc, bg, fg in stages:
    box = patches.FancyBboxPatch((0.9, y_pos - 0.58), 6.2, 0.68, boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor=bg, edgecolor=fg, lw=1.2)
    ax.add_patch(box)
    ax.text(1.1, y_pos - 0.15, name, color=fg, fontsize=9.2, fontweight='bold', va='center')
    ax.text(1.1, y_pos - 0.42, desc, color="#334155", fontsize=7.8, va='center')
    
    if y_pos > 1.8:
        ax.annotate('', xy=(4.0, y_pos - 0.65), xytext=(4.0, y_pos - 0.58),
                    arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.5))
    y_pos -= 0.82

# ==============================================================================
# ฝั่งขวา: หัวใจสำคัญ - Residual Bottleneck Block & Skip Connection
# ==============================================================================
card_right = patches.FancyBboxPatch((7.7, 1.0), 6.7, 6.3, boxstyle="round,pad=0.08,rounding_size=0.15",
                                   facecolor="#FFFFFF", edgecolor="#10B981", linewidth=1.8)
ax.add_patch(card_right)
ax.text(11.05, 7.0, "2. หัวใจสำคัญ: Residual Block (Skip Connection)", color="#059669", fontsize=12, fontweight='bold', ha='center')

# แผนผัง Bottleneck Block
ax.text(10.0, 6.35, "Input: x", fontsize=11, fontweight='bold', color="#0F172A", ha='center')
ax.plot([10.0, 10.0], [6.15, 5.8], color="#0F172A", lw=2)

# Box 1: 1x1 Conv
b1 = patches.Rectangle((8.8, 5.25), 2.4, 0.55, facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.5)
ax.add_patch(b1)
ax.text(10.0, 5.52, "1×1 Conv, ReLU\n(ลดมิติช่องสัญญาณ)", color="#1E40AF", fontsize=8.5, ha='center', va='center')

ax.annotate('', xy=(10.0, 4.8), xytext=(10.0, 5.25),
            arrowprops=dict(arrowstyle="->", color="#0F172A", lw=2))

# Box 2: 3x3 Conv
b2 = patches.Rectangle((8.8, 4.25), 2.4, 0.55, facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.5)
ax.add_patch(b2)
ax.text(10.0, 4.52, "3×3 Conv, ReLU\n(สกัดฟีเจอร์รอยโรค)", color="#1E40AF", fontsize=8.5, ha='center', va='center')

ax.annotate('', xy=(10.0, 3.8), xytext=(10.0, 4.25),
            arrowprops=dict(arrowstyle="->", color="#0F172A", lw=2))

# Box 3: 1x1 Conv
b3 = patches.Rectangle((8.8, 3.25), 2.4, 0.55, facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.5)
ax.add_patch(b3)
ax.text(10.0, 3.52, "1×1 Conv\n(ขยายมิติกลับคืน)", color="#1E40AF", fontsize=8.5, ha='center', va='center')

# วงกลม Addition (+)
add_circle = plt.Circle((10.0, 2.45), 0.28, facecolor="#FEF3C7", edgecolor="#D97706", lw=2)
ax.add_patch(add_circle)
ax.text(10.0, 2.45, "+", color="#92400E", fontsize=15, fontweight='bold', ha='center', va='center')

ax.annotate('', xy=(10.0, 2.75), xytext=(10.0, 3.25),
            arrowprops=dict(arrowstyle="->", color="#0F172A", lw=2))

# เส้น Skip Connection (Shortcut) อ้อมจาก x ข้ามมาที่ (+)
ax.plot([10.0, 12.8, 12.8, 10.3], [6.0, 6.0, 2.45, 2.45], color="#DC2626", lw=2.5, linestyle="--")
ax.annotate('', xy=(10.3, 2.45), xytext=(10.5, 2.45),
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=2.5))
ax.text(13.0, 4.3, "Skip Connection\n(ทางลัดส่งข้อมูลข้ามชั้น)\nIdentity: x", color="#B91C1C", fontsize=9, fontweight='bold', ha='left', va='center')

# Output F(x) + x
ax.annotate('', xy=(10.0, 1.85), xytext=(10.0, 2.15),
            arrowprops=dict(arrowstyle="->", color="#0F172A", lw=2))
ax.text(10.0, 1.65, r"Output: $\mathcal{F}(x) + x$ (ผ่าน ReLU)", fontsize=11, fontweight='bold', color="#065F46", ha='center')

# คำอธิบายประโยชน์ของ Skip Connection ด้านล่าง
exp_box = patches.FancyBboxPatch((8.0, 1.15), 6.1, 0.45, boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor="#F0FDF4", edgecolor="#86EFAC", lw=1.2)
ax.add_patch(exp_box)
ax.text(11.05, 1.37, "แก้ปัญหา Vanishing Gradient ทำให้เทรนโครงข่ายลึก 50 เลเยอร์ได้โดยค่าน้ำหนักไม่พัง",
        color="#15803D", fontsize=8.2, fontweight='bold', ha='center', va='center')

# Footer
ax.text(7.5, 0.45, "โมบายแอปพลิเคชันประเมินความเสี่ยงมะเร็งผิวหนังและรอยโรคทั่วไป | วิศวกรรมคอมพิวเตอร์ มทร.อีสาน",
        color="#64748B", fontsize=9.5, ha='center')

out_path = "resnet50_architecture.png"
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Generated: {out_path}")
