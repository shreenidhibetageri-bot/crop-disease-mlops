import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# ── Path Settings ────────────────────────────────────────────
DATA_PATH = "data/raw/plantvillage dataset/color"
SAVE_PATH = "notebooks/eda_results"
os.makedirs(SAVE_PATH, exist_ok=True)

print("Starting EDA on PlantVillage Dataset...")
print("=" * 50)

# ── Step 1: Count Images per Class ───────────────────────────
print("\n📊 Step 1: Counting images per class...")
class_counts = {}
for class_name in sorted(os.listdir(DATA_PATH)):
    class_folder = os.path.join(DATA_PATH, class_name)
    if os.path.isdir(class_folder):
        count = len(os.listdir(class_folder))
        class_counts[class_name] = count
        print(f"  {class_name}: {count} images")

print(f"\nTotal classes: {len(class_counts)}")
print(f"Total images: {sum(class_counts.values())}")

# Plot class distribution
plt.figure(figsize=(22, 8))
classes = list(class_counts.keys())
counts = list(class_counts.values())
short_names = [c.replace("___", "\n").replace("_", " ")[:20] for c in classes]
bars = plt.bar(range(len(classes)), counts,
               color=['#1d9e75' if 'healthy' in c.lower()
                      else '#e24b4a' for c in classes])
plt.xticks(range(len(classes)), short_names,
           rotation=90, fontsize=7)
plt.title("Number of Images per Disease Class\n(Green = Healthy, Red = Diseased)",
          fontsize=14, fontweight='bold')
plt.xlabel("Disease Class", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/1_class_distribution.png", dpi=150)
plt.close()
print("✅ Class distribution chart saved!")

# ── Step 2: Healthy vs Diseased ──────────────────────────────
print("\n🌿 Step 2: Healthy vs Diseased analysis...")
healthy_count = 0
diseased_count = 0
healthy_classes = 0
diseased_classes = 0

for class_name, count in class_counts.items():
    if "healthy" in class_name.lower():
        healthy_count += count
        healthy_classes += 1
    else:
        diseased_count += count
        diseased_classes += 1

print(f"  Healthy images: {healthy_count} ({healthy_classes} classes)")
print(f"  Diseased images: {diseased_count} ({diseased_classes} classes)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart - images
ax1.pie([healthy_count, diseased_count],
        labels=[f'Healthy\n({healthy_count} images)',
                f'Diseased\n({diseased_count} images)'],
        colors=['#1d9e75', '#e24b4a'],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12})
ax1.set_title("Image Distribution\nHealthy vs Diseased",
              fontsize=13, fontweight='bold')

# Bar chart - classes
ax2.bar(['Healthy Classes', 'Diseased Classes'],
        [healthy_classes, diseased_classes],
        color=['#1d9e75', '#e24b4a'], width=0.4)
ax2.set_title("Number of Classes\nHealthy vs Diseased",
              fontsize=13, fontweight='bold')
ax2.set_ylabel("Number of Classes")
for i, v in enumerate([healthy_classes, diseased_classes]):
    ax2.text(i, v + 0.1, str(v), ha='center',
             fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/2_healthy_vs_diseased.png", dpi=150)
plt.close()
print("✅ Healthy vs Diseased chart saved!")

# ── Step 3: Sample Images ────────────────────────────────────
print("\n🖼️  Step 3: Visualizing sample images...")
sample_classes = random.sample(list(class_counts.keys()), min(20, len(class_counts)))

fig, axes = plt.subplots(4, 5, figsize=(18, 14))
fig.suptitle("Sample Images from Different Disease Classes",
             fontsize=16, fontweight='bold', y=1.02)

for i, ax in enumerate(axes.flat):
    if i < len(sample_classes):
        class_name = sample_classes[i]
        class_folder = os.path.join(DATA_PATH, class_name)
        img_files = os.listdir(class_folder)
        img_path = os.path.join(class_folder, random.choice(img_files))
        try:
            img = mpimg.imread(img_path)
            ax.imshow(img)
            short = class_name.replace("___", "\n").replace("_", " ")
            color = '#1d9e75' if 'healthy' in class_name.lower() else '#e24b4a'
            ax.set_title(short[:25], fontsize=7, color=color, fontweight='bold')
        except:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax.axis('off')

plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/3_sample_images.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Sample images chart saved!")

# ── Step 4: Image Size Analysis ──────────────────────────────
print("\n📐 Step 4: Checking image sizes...")
widths = []
heights = []
sample_classes_check = list(class_counts.keys())[:10]

for class_name in sample_classes_check:
    class_folder = os.path.join(DATA_PATH, class_name)
    img_files = os.listdir(class_folder)[:20]
    for img_file in img_files:
        try:
            img = Image.open(os.path.join(class_folder, img_file))
            w, h = img.size
            widths.append(w)
            heights.append(h)
        except:
            pass

print(f"  Most common width: {max(set(widths), key=widths.count)}px")
print(f"  Most common height: {max(set(heights), key=heights.count)}px")
print(f"  Min size: {min(widths)}x{min(heights)}")
print(f"  Max size: {max(widths)}x{max(heights)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(widths, bins=20, color='#1d9e75', edgecolor='black')
ax1.set_title("Image Width Distribution", fontweight='bold')
ax1.set_xlabel("Width (pixels)")
ax1.set_ylabel("Count")

ax2.hist(heights, bins=20, color='#0f6e56', edgecolor='black')
ax2.set_title("Image Height Distribution", fontweight='bold')
ax2.set_xlabel("Height (pixels)")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/4_image_sizes.png", dpi=150)
plt.close()
print("✅ Image size chart saved!")

# ── Step 5: Pixel Intensity Analysis ─────────────────────────
print("\n🎨 Step 5: Analyzing pixel intensities...")
means_r, means_g, means_b = [], [], []
sample_classes_pixel = list(class_counts.keys())[:8]

for class_name in sample_classes_pixel:
    class_folder = os.path.join(DATA_PATH, class_name)
    img_files = random.sample(os.listdir(class_folder),
                              min(10, len(os.listdir(class_folder))))
    for img_file in img_files:
        try:
            img = np.array(Image.open(
                os.path.join(class_folder, img_file)).convert('RGB'))
            means_r.append(img[:,:,0].mean())
            means_g.append(img[:,:,1].mean())
            means_b.append(img[:,:,2].mean())
        except:
            pass

plt.figure(figsize=(10, 5))
plt.hist(means_r, bins=30, alpha=0.7, color='red', label='Red Channel')
plt.hist(means_g, bins=30, alpha=0.7, color='green', label='Green Channel')
plt.hist(means_b, bins=30, alpha=0.7, color='blue', label='Blue Channel')
plt.title("RGB Pixel Intensity Distribution", fontsize=13, fontweight='bold')
plt.xlabel("Mean Pixel Value (0-255)")
plt.ylabel("Number of Images")
plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/5_pixel_intensity.png", dpi=150)
plt.close()
print("✅ Pixel intensity chart saved!")

# ── Step 6: Top 10 Largest and Smallest Classes ──────────────
print("\n📈 Step 6: Top and bottom classes by image count...")
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
top10 = sorted_classes[:10]
bottom10 = sorted_classes[-10:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Top 10
names_top = [c[0].replace("___", "\n").replace("_"," ")[:20] for c in top10]
counts_top = [c[1] for c in top10]
ax1.barh(names_top, counts_top, color='#1d9e75')
ax1.set_title("Top 10 Classes (Most Images)", fontweight='bold')
ax1.set_xlabel("Number of Images")
for i, v in enumerate(counts_top):
    ax1.text(v + 10, i, str(v), va='center', fontsize=9)

# Bottom 10
names_bot = [c[0].replace("___", "\n").replace("_"," ")[:20] for c in bottom10]
counts_bot = [c[1] for c in bottom10]
ax2.barh(names_bot, counts_bot, color='#e24b4a')
ax2.set_title("Bottom 10 Classes (Least Images)", fontweight='bold')
ax2.set_xlabel("Number of Images")
for i, v in enumerate(counts_bot):
    ax2.text(v + 10, i, str(v), va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/6_top_bottom_classes.png", dpi=150)
plt.close()
print("✅ Top/Bottom classes chart saved!")

# ── Final Summary ────────────────────────────────────────────
print("\n" + "=" * 50)
print("📊 EDA COMPLETE! SUMMARY REPORT")
print("=" * 50)
print(f"Total Classes      : {len(class_counts)}")
print(f"Total Images       : {sum(class_counts.values())}")
print(f"Healthy Classes    : {healthy_classes}")
print(f"Diseased Classes   : {diseased_classes}")
print(f"Healthy Images     : {healthy_count}")
print(f"Diseased Images    : {diseased_count}")
print(f"Most Images Class  : {sorted_classes[0][0]} ({sorted_classes[0][1]})")
print(f"Least Images Class : {sorted_classes[-1][0]} ({sorted_classes[-1][1]})")
print(f"\nAll charts saved in: notebooks/eda_results/")
print("\nFiles saved:")
for f in os.listdir(SAVE_PATH):
    print(f"  ✅ {f}")