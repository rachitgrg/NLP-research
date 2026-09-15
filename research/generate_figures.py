import os
import csv
import json
import statistics
import matplotlib.pyplot as plt
import matplotlib.patches as patches

base_dir = os.path.dirname(os.path.abspath(__file__))
research_dir = os.path.join(base_dir, 'research')
figures_dir = os.path.join(research_dir, 'figures')

os.makedirs(figures_dir, exist_ok=True)

# 1. READ CANONICAL DATA
results_path = os.path.join(base_dir, 'module-1', 'evaluation', 'results.csv')
with open(results_path, 'r', encoding='utf-8') as f:
    results_rows = list(csv.DictReader(f))

models = [('sarvam', 'Sarvam AI'), ('whisper_tiny', 'Whisper Tiny'), ('whisper_base', 'Whisper Base')]

asr_aggregates = {}
for m_key, m_name in models:
    wers = [float(r[f'{m_key}_wer']) for r in results_rows]
    cers = [float(r[f'{m_key}_cer']) for r in results_rows]
    ems = [1.0 if r[f'{m_key}_exact_match'] == 'True' else 0.0 for r in results_rows]
    lats = [float(r[f'{m_key}_latency_seconds']) for r in results_rows]
    
    asr_aggregates[m_key] = {
        'name': m_name,
        'wer': statistics.mean(wers),
        'cer': statistics.mean(cers),
        'em': statistics.mean(ems),
        'lat_mean': statistics.mean(lats),
        'lat_median': statistics.median(lats)
    }

semantic_data = {
    'Reference': {'intent': 100.0, 'object': 100.0, 'attribute': 100.0, 'complete': 100.0},
    'Sarvam AI': {'intent': 100.0, 'object': 100.0, 'attribute': 100.0, 'complete': 100.0},
    'Whisper Tiny': {'intent': 100.0, 'object': 100.0, 'attribute': 100.0, 'complete': 100.0},
    'Whisper Base': {'intent': 90.0, 'object': 100.0, 'attribute': 100.0, 'complete': 90.0}
}

refcoco_spatial = {'left': 63060, 'right': 62326, 'front': 15814, 'bottom': 13704, 'middle': 12622, 'top': 12260, 'far': 10198, 'back': 8426}
refcoco_color = {'white': 14510, 'black': 10604, 'blue': 10574, 'red': 9894, 'green': 5072, 'yellow': 3386, 'orange': 3270, 'brown': 3212}
refcoco_size = {'big': 1980, 'little': 1172, 'small': 910, 'tall': 524, 'large': 460, 'long': 426, 'short': 256}

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def plot_bar(x_labels, values, title, ylabel, filename, color='#4A90D9', is_percent=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x_labels, values, color=color, width=0.6)
    ax.set_title(title, pad=15)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        h = bar.get_height()
        fmt = f'{h:.1f}%' if is_percent else f'{h:.4f}'
        if max(values) > 100:
            fmt = f'{int(h):,}'
        ax.text(bar.get_x() + bar.get_width()/2., h + (max(values)*0.02), fmt,
                ha='center', va='bottom', fontweight='bold')
                
    if is_percent:
        ax.set_ylim(0, max(values) * 1.15)
        
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300)
    plt.close()

model_names = [a['name'] for a in asr_aggregates.values()]

# FIG 2-5: ASR Metrics
plot_bar(model_names, [a['wer']*100 for a in asr_aggregates.values()], 'ASR Word Error Rate (WER)', 'WER (%)', 'fig_02_asr_wer.png', color='#E05C3A', is_percent=True)
plot_bar(model_names, [a['cer']*100 for a in asr_aggregates.values()], 'ASR Character Error Rate (CER)', 'CER (%)', 'fig_03_asr_cer.png', color='#E05C3A', is_percent=True)
plot_bar(model_names, [a['em']*100 for a in asr_aggregates.values()], 'ASR Exact Match Rate', 'Exact Match (%)', 'fig_04_asr_exact_match.png', color='#56B4E9', is_percent=True)
plot_bar(model_names, [a['lat_mean'] for a in asr_aggregates.values()], 'Mean Transcription Latency', 'Latency (s)', 'fig_05_asr_latency.png', color='#F39C12')

# FIG 6: Accuracy vs Latency
fig, ax = plt.subplots(figsize=(8, 6))
x = [a['lat_mean'] for a in asr_aggregates.values()]
y = [100.0 - (a['wer']*100) for a in asr_aggregates.values()] # 1-WER approx for accuracy in this plot
colors = ['#E05C3A', '#4A90D9', '#56B4E9']
for i, txt in enumerate(model_names):
    ax.scatter(x[i], y[i], color=colors[i], s=200, label=txt, zorder=5)
    ax.annotate(txt, (x[i], y[i]), xytext=(10, -5), textcoords='offset points', fontweight='bold')
ax.set_title('Transcription Accuracy vs. Mean Latency', pad=15)
ax.set_xlabel('Mean Latency (s)')
ax.set_ylabel('Accuracy (100 - WER%)')
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'fig_06_accuracy_latency_tradeoff.png'), dpi=300)
plt.close()

# FIG 7: Semantic Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
import numpy as np
sem_labels = ['Intent', 'Object', 'Attribute', 'Complete']
x = np.arange(len(sem_labels))
width = 0.2
models_sem = list(semantic_data.keys())
sem_colors = ['#2C3E50', '#E05C3A', '#4A90D9', '#56B4E9']

for i, model in enumerate(models_sem):
    vals = [semantic_data[model]['intent'], semantic_data[model]['object'], semantic_data[model]['attribute'], semantic_data[model]['complete']]
    ax.bar(x + i*width, vals, width, label=model, color=sem_colors[i])

ax.set_ylabel('Accuracy (%)')
ax.set_title('Semantic Preservation Accuracy', pad=15)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(sem_labels)
ax.legend(loc='lower right')
ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'fig_07_semantic_accuracy.png'), dpi=300)
plt.close()

# FIG 8-10: Dataset EDA
def plot_horizontal(data_dict, title, xlabel, filename, color):
    items = sorted(data_dict.items(), key=lambda x: x[1])
    labels = [k for k,v in items]
    vals = [v for k,v in items]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(labels, vals, color=color)
    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + (max(vals)*0.01), bar.get_y() + bar.get_height()/2, f'{int(w):,}',
                ha='left', va='center')
                
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300)
    plt.close()

plot_horizontal(refcoco_spatial, 'RefCOCO Spatial Language Frequency', 'Occurrences', 'fig_08_refcoco_spatial_terms.png', '#8E44AD')
plot_horizontal(refcoco_color, 'RefCOCO Color Term Frequency', 'Occurrences', 'fig_09_refcoco_color_terms.png', '#27AE60')
plot_horizontal(refcoco_size, 'RefCOCO Size Term Frequency', 'Occurrences', 'fig_10_refcoco_size_terms.png', '#D35400')

# FIG 11 & 12: Imbalance
def plot_pie(sizes, labels, title, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#E74C3C', '#BDC3C7'])
    ax.set_title(title)
    plt.savefig(os.path.join(figures_dir, filename), dpi=300)
    plt.close()

plot_pie([185315, 604906 - 185315], ['Person', 'Other (79 classes)'], 'COCO Category Distribution (Annotations)', 'fig_11_coco_category_distribution.png')
plot_pie([134252, 267568 - 134252], ['Person', 'Other (77 classes)'], 'RefCOCO Category Distribution (Expressions)', 'fig_12_refcoco_category_distribution.png')

# FIG 1 & 13: Architecture Diagrams (using blocks)
def create_diagram(blocks, arrows, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    for (x, y, w, h, text) in blocks:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='#34495E', facecolor='#ECF0F1')
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=11, fontweight='bold')
        
    for (start, end, label) in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(facecolor='#2C3E50', width=2, headwidth=8))
        if label:
            mx = (start[0]+end[0])/2
            my = (start[1]+end[1])/2
            ax.text(mx, my+0.3, label, ha='center', va='center', fontsize=9, color='#C0392B')
            
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    plt.savefig(os.path.join(figures_dir, filename), dpi=300)
    plt.close()

stage1_blocks = [
    (1, 6, 2, 1, 'Microphone'),
    (1, 4, 2, 1, 'ASR Model\n(Sarvam/Whisper)'),
    (1, 2, 2, 1, 'NLP Parser\n(spaCy)'),
    (4, 2, 2, 1, 'Semantic Output\n(Intent/Object)'),
    (7, 2, 2, 1, 'Visual Grounding\n(Future Stage)')
]
stage1_arrows = [
    ((2, 6), (2, 5), 'Audio'),
    ((2, 4), (2, 3), 'English Text'),
    ((3, 2.5), (4, 2.5), ''),
    ((6, 2.5), (7, 2.5), 'Target Info')
]
create_diagram(stage1_blocks, stage1_arrows, 'Stage 1 System Architecture', 'fig_01_stage1_architecture.png')

ds_blocks = [
    (1, 6, 3, 1, 'COCO + RefCOCO\n(Raw Datasets)'),
    (1, 4, 3, 1, 'Validation & Cleaning'),
    (1, 2, 3, 1, 'Integrity Checking & EDA'),
    (6, 2, 3, 1, 'Visual Grounding Dataset\n(Future Stage)')
]
ds_arrows = [
    ((2.5, 6), (2.5, 5), ''),
    ((2.5, 4), (2.5, 3), ''),
    ((4, 2.5), (6, 2.5), 'Prepared Data')
]
create_diagram(ds_blocks, ds_arrows, 'Dataset Preprocessing Pipeline', 'fig_13_dataset_pipeline.png')

print("Figures generated successfully.")
