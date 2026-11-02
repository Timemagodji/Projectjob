import matplotlib.pyplot as plt
import numpy as np

# ตั้งค่าฟอนต์ภาษาไทย
plt.rcParams['font.family'] = 'Leelawadee UI'
plt.rcParams['axes.unicode_minus'] = False

classes_vert = [
    "Melanoma\n(มะเร็งเมลาโนมา)",
    "Melanocytic Nevi\n(ไฝ / ปาน)",
    "Keratosis\n(กระเนื้อ)",
    "BCC\n(มะเร็งเบซัลเซลล์)",
    "SCC\n(มะเร็งสความัสเซลล์)",
    "Actinic Keratosis\n(ระยะก่อนมะเร็ง)",
    "Psoriasis\n(สะเก็ดเงิน)",
    "Tinea\n(เชื้อราผิวหนัง)"
]

classes_single_line = [
    "Melanoma (มะเร็งเมลาโนมา)",
    "Melanocytic Nevi (ไฝ/ปาน)",
    "Keratosis (กระเนื้อ)",
    "BCC (มะเร็งเบซัลเซลล์)",
    "SCC (มะเร็งสความัสเซลล์)",
    "Actinic Keratosis (ระยะก่อนมะเร็ง)",
    "Psoriasis (สะเก็ดเงิน)",
    "Tinea (เชื้อราผิวหนัง)"
]

total_counts = [8183, 8170, 4143, 3323, 3031, 2808, 2055, 1906]
percentages = [24.34, 24.30, 12.32, 9.88, 9.02, 8.35, 6.11, 5.67]

train_counts = [5728, 5719, 2900, 2326, 2121, 1965, 1438, 1334]
val_counts   = [1636, 1634,  828,  664,  606,  561,  411,  381]
test_counts  = [ 819,  817,  415,  333,  304,  282,  206,  191]

y_pos = np.arange(len(classes_single_line))
y_pos_rev = y_pos[::-1]
classes_rev = classes_single_line[::-1]
total_rev = total_counts[::-1]
pct_rev = percentages[::-1]

# ==============================================================================
# กราฟที่ 1: กราฟแท่งแสดงจำนวนรูปข้อมูลของแต่ละโรค (Total Images per Disease)
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(12, 7), dpi=300, facecolor='#F8FAFC')
ax1.set_facecolor('#FFFFFF')

colors1 = ['#1D4ED8', '#2563EB', '#3B82F6', '#60A5FA', '#818CF8', '#A78BFA', '#0D9488', '#14B8A6']
colors_rev = colors1[::-1]

bars1 = ax1.barh(y_pos_rev, total_rev, color=colors_rev, height=0.62, edgecolor='#CBD5E1', linewidth=1, zorder=3)

for bar, count, pct in zip(bars1, total_rev, pct_rev):
    width = bar.get_width()
    ax1.text(width + 120, bar.get_y() + bar.get_height()/2,
             f"{count:,} ภาพ  ({pct:.2f}%)",
             va='center', ha='left', fontsize=10.5, fontweight='bold', color='#1E293B', zorder=4)

ax1.set_yticks(y_pos_rev)
ax1.set_yticklabels(classes_rev, fontsize=11, fontweight='bold', color='#334155')
ax1.set_xlabel("จำนวนภาพถ่าย (Images)", fontsize=11, fontweight='bold', color='#475569', labelpad=10)
ax1.set_xlim(0, 10000)
ax1.set_title("จำนวนภาพข้อมูลในแต่ละกลุ่มโรคผิวหนัง (รวมทั้งหมด 33,619 ภาพ)",
              fontsize=15, fontweight='bold', color='#0F172A', pad=15)

ax1.grid(axis='x', linestyle='--', alpha=0.5, color='#94A3B8', zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#CBD5E1')
ax1.spines['bottom'].set_color('#CBD5E1')

plt.tight_layout()
fig1_path = "graph1_dataset_per_disease.png"
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)
print(f"Generated: {fig1_path}")


# ==============================================================================
# กราฟที่ 2: กราฟแท่งกลุ่ม (Grouped Bar Chart) แสดง Train / Val / Test (70% : 20% : 10%)
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(14, 7.5), dpi=300, facecolor='#F8FAFC')
ax2.set_facecolor('#FFFFFF')

x = np.arange(len(classes_vert))
width = 0.26

rects1 = ax2.bar(x - width, train_counts, width, label='Train Set (70%)', color='#2563EB', edgecolor='#1D4ED8', linewidth=1, zorder=3)
rects2 = ax2.bar(x,         val_counts,   width, label='Validation Set (20%)', color='#F59E0B', edgecolor='#D97706', linewidth=1, zorder=3)
rects3 = ax2.bar(x + width, test_counts,  width, label='Test Set (10%)', color='#10B981', edgecolor='#059669', linewidth=1, zorder=3)

for rect in rects1:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + 75, f'{h:,}',
             ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E40AF')

for rect in rects2:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + 75, f'{h:,}',
             ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#B45309')

for rect in rects3:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + 75, f'{h:,}',
             ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#065F46')

ax2.set_xticks(x)
ax2.set_xticklabels(classes_vert, fontsize=9.5, fontweight='bold', color='#1E293B')
ax2.set_ylabel("จำนวนภาพถ่าย (Images)", fontsize=11, fontweight='bold', color='#475569', labelpad=10)
ax2.set_ylim(0, 6500)
ax2.set_title("จำนวนภาพในการจัดสรรชุดข้อมูลสำหรับฝึกสอนโมเดล ResNet-50 แยกตามรายโรค (70% : 20% : 10%)",
              fontsize=14, fontweight='bold', color='#0F172A', pad=15)

legend = ax2.legend(loc='upper right', frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=11)
legend.get_texts()[0].set_color('#1E40AF')
legend.get_texts()[1].set_color('#B45309')
legend.get_texts()[2].set_color('#065F46')

ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#94A3B8', zorder=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#CBD5E1')
ax2.spines['bottom'].set_color('#CBD5E1')

plt.tight_layout()
fig2_path = "graph2_dataset_split_70_20_10.png"
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)
print(f"Generated: {fig2_path}")


# ==============================================================================
# กราฟที่ 2 (เวอร์ชัน Stacked Bar แนวนอน: เห็นยอดรวมและสัดส่วนครบในแถวเดียว)
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=(13.5, 7.2), dpi=300, facecolor='#F8FAFC')
ax3.set_facecolor('#FFFFFF')

bars_train = ax3.barh(y_pos_rev, train_counts[::-1], color='#2563EB', height=0.58, label='Train Set (70%)', edgecolor='#CBD5E1', zorder=3)
bars_val   = ax3.barh(y_pos_rev, val_counts[::-1],   left=train_counts[::-1], color='#F59E0B', height=0.58, label='Validation Set (20%)', edgecolor='#CBD5E1', zorder=3)
left_test  = np.array(train_counts[::-1]) + np.array(val_counts[::-1])
bars_test  = ax3.barh(y_pos_rev, test_counts[::-1],  left=left_test, color='#10B981', height=0.58, label='Test Set (10%)', edgecolor='#CBD5E1', zorder=3)

for i in range(len(classes_rev)):
    tr = train_counts[::-1][i]
    va = val_counts[::-1][i]
    te = test_counts[::-1][i]
    tot = total_counts[::-1][i]
    y = y_pos_rev[i]

    # ตัวเลขใน Train
    ax3.text(tr / 2, y, f"{tr:,}", ha='center', va='center', color='#FFFFFF', fontsize=9.5, fontweight='bold', zorder=4)
    # ตัวเลขใน Val
    ax3.text(tr + va / 2, y, f"{va:,}", ha='center', va='center', color='#FFFFFF', fontsize=8.5, fontweight='bold', zorder=4)
    # ตัวเลขใน Test
    ax3.text(tr + va + te / 2, y, f"{te:,}", ha='center', va='center', color='#FFFFFF', fontsize=8, fontweight='bold', zorder=4)
    # ยอดรวมหลังแท่ง
    ax3.text(tot + 140, y, f"รวม {tot:,} ภาพ", ha='left', va='center', color='#0F172A', fontsize=10, fontweight='bold', zorder=4)

ax3.set_yticks(y_pos_rev)
ax3.set_yticklabels(classes_rev, fontsize=10.5, fontweight='bold', color='#334155')
ax3.set_xlabel("จำนวนภาพถ่าย (Images)", fontsize=11, fontweight='bold', color='#475569', labelpad=10)
ax3.set_xlim(0, 10200)
ax3.set_title("สัดส่วนการจัดสรรชุดข้อมูล Train (70%) : Validation (20%) : Test (10%) ในแต่ละโรค",
              fontsize=14, fontweight='bold', color='#0F172A', pad=15)

# วาง Legend มุมบนขวา
ax3.legend(loc='upper right', frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=10.5)
ax3.grid(axis='x', linestyle='--', alpha=0.5, color='#94A3B8', zorder=0)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color('#CBD5E1')
ax3.spines['bottom'].set_color('#CBD5E1')

plt.tight_layout()
fig3_path = "graph2_dataset_split_stacked.png"
fig3.savefig(fig3_path, dpi=300)
plt.close(fig3)
print(f"Generated: {fig3_path}")


# ==============================================================================
# กราฟรวม 2-in-1 สำหรับสไลด์ PowerPoint แผ่นเดียว (Combined Slide)
# ==============================================================================
fig_all, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=300, facecolor='#F8FAFC')

# --- ซ้าย: กราฟภาพรวมแต่ละโรค ---
ax_left.set_facecolor('#FFFFFF')
bars_l = ax_left.barh(y_pos_rev, total_rev, color=colors_rev, height=0.62, edgecolor='#CBD5E1', zorder=3)
for bar, count, pct in zip(bars_l, total_rev, pct_rev):
    width = bar.get_width()
    ax_left.text(width + 100, bar.get_y() + bar.get_height()/2,
                 f"{count:,} ({pct:.1f}%)",
                 va='center', ha='left', fontsize=9.5, fontweight='bold', color='#1E293B', zorder=4)
ax_left.set_yticks(y_pos_rev)
ax_left.set_yticklabels(classes_rev, fontsize=10, fontweight='bold', color='#334155')
ax_left.set_xlabel("จำนวนภาพถ่าย (Images)", fontsize=10, fontweight='bold', color='#475569')
ax_left.set_xlim(0, 10200)
ax_left.set_title("(ก) จำนวนภาพข้อมูลรวมในแต่ละโรค (ทั้งหมด 33,619 ภาพ)", fontsize=12.5, fontweight='bold', color='#0F172A', pad=12)
ax_left.grid(axis='x', linestyle='--', alpha=0.5, color='#94A3B8', zorder=0)
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)

# --- ขวา: กราฟแท่งกลุ่ม Train / Val / Test ---
ax_right.set_facecolor('#FFFFFF')
r1 = ax_right.bar(x - width, train_counts, width, label='Train (70%)', color='#2563EB', zorder=3)
r2 = ax_right.bar(x,         val_counts,   width, label='Val (20%)', color='#F59E0B', zorder=3)
r3 = ax_right.bar(x + width, test_counts,  width, label='Test (10%)', color='#10B981', zorder=3)

for rect in r1:
    h = rect.get_height()
    ax_right.text(rect.get_x() + rect.get_width()/2., h + 70, f'{h:,}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1E40AF')
for rect in r2:
    h = rect.get_height()
    ax_right.text(rect.get_x() + rect.get_width()/2., h + 70, f'{h:,}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#B45309')
for rect in r3:
    h = rect.get_height()
    ax_right.text(rect.get_x() + rect.get_width()/2., h + 70, f'{h:,}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#065F46')

ax_right.set_xticks(x)
ax_right.set_xticklabels(classes_vert, fontsize=8.5, fontweight='bold', color='#1E293B')
ax_right.set_ylabel("จำนวนภาพถ่าย (Images)", fontsize=10, fontweight='bold', color='#475569')
ax_right.set_ylim(0, 6500)
ax_right.set_title("(ข) การจัดสรรชุดข้อมูลแยกรายโรค (Train 70% : Val 20% : Test 10%)", fontsize=12.5, fontweight='bold', color='#0F172A', pad=12)
ax_right.legend(loc='upper right', frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=9.5)
ax_right.grid(axis='y', linestyle='--', alpha=0.5, color='#94A3B8', zorder=0)
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)

plt.tight_layout()
fig_all_path = "dataset_charts_combined.png"
fig_all.savefig(fig_all_path, dpi=300)
plt.close(fig_all)
print(f"Generated: {fig_all_path}")
