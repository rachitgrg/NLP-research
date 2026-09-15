import os
import csv
import json
import statistics

base_dir = os.path.dirname(os.path.abspath(__file__))
research_dir = os.path.join(base_dir, 'research')
tables_dir = os.path.join(research_dir, 'tables')
figures_dir = os.path.join(research_dir, 'figures')

os.makedirs(tables_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# 1. READ CANONICAL DATA
results_path = os.path.join(base_dir, 'module-1', 'evaluation', 'results.csv')
with open(results_path, 'r', encoding='utf-8') as f:
    results_rows = list(csv.DictReader(f))

# Recalculate aggregates from results.csv strictly to match constraints
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

# Semantic Data is currently verified externally, but we will hardcode the verified semantic numbers as per instructions:
semantic_data = {
    'Reference': {'intent': 1.0, 'object': 1.0, 'attribute': 1.0, 'complete': 1.0},
    'Sarvam': {'intent': 1.0, 'object': 1.0, 'attribute': 1.0, 'complete': 1.0},
    'Whisper Tiny': {'intent': 1.0, 'object': 1.0, 'attribute': 1.0, 'complete': 1.0},
    'Whisper Base': {'intent': 0.9, 'object': 1.0, 'attribute': 1.0, 'complete': 0.9}
}

# COCO / RefCOCO data (from instructions/verified dataset-analysis outputs)
coco_stats = {
    'Source Images': 82783,
    'Original Annotations': 604907,
    'Valid Annotations': 604906,
    'Categories': 80,
    'Unique Image IDs': 82081,
    'Person Annotations': 185315
}

refcoco_stats = {
    'UNC References': 50000,
    'Google References': 50000,
    'Expressions Before Dedup': 284420,
    'Duplicate Expressions': 16852,
    'Final Expressions': 267568,
    'Unique Images': 19994,
    'Unique Annotation Targets': 50000,
    'Categories': 78,
    'Referential Mismatches': 0,
    'Person Expressions': 134252
}

refcoco_linguistic = {
    'Tokens': 961454,
    'Unique Words': 10116,
    'Average Length': 3.59,
    'Median Length': 3
}

refcoco_spatial = {'left': 63060, 'right': 62326, 'front': 15814, 'bottom': 13704, 'middle': 12622, 'top': 12260, 'far': 10198, 'back': 8426}
refcoco_color = {'white': 14510, 'black': 10604, 'blue': 10574, 'red': 9894, 'green': 5072, 'yellow': 3386, 'orange': 3270, 'brown': 3212}
refcoco_size = {'big': 1980, 'little': 1172, 'small': 910, 'tall': 524, 'large': 460, 'long': 426, 'short': 256}

# 2. GENERATE TABLES
def write_md_table(filename, headers, rows):
    path = os.path.join(tables_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('|' + '|'.join(['---' for _ in headers]) + '|\n')
        for row in rows:
            f.write('| ' + ' | '.join([str(x) for x in row]) + ' |\n')

# TABLE 1: ASR
write_md_table('table_01_asr_comparison.md', 
    ['Model', 'WER (%)', 'CER (%)', 'Exact Match (%)', 'Mean Latency (s)', 'Median Latency (s)'],
    [[agg['name'], f"{agg['wer']*100:.2f}", f"{agg['cer']*100:.2f}", f"{agg['em']*100:.2f}", f"{agg['lat_mean']:.4f}", f"{agg['lat_median']:.4f}"] for agg in asr_aggregates.values()]
)

# TABLE 2: Semantic
write_md_table('table_02_semantic_eval.md',
    ['Model', 'Intent Accuracy', 'Object Accuracy', 'Attribute Accuracy', 'Complete Semantic Accuracy'],
    [[model, f"{data['intent']*100:.0f}%", f"{data['object']*100:.0f}%", f"{data['attribute']*100:.0f}%", f"{data['complete']*100:.0f}%"] for model, data in semantic_data.items()]
)

# TABLE 3: Individual ASR
table_3_rows = []
for r in results_rows:
    table_3_rows.append([
        r['sample_id'], 
        r['reference'], 
        r['sarvam_text'], 
        r['whisper_tiny_text'], 
        r['whisper_base_text']
    ])
write_md_table('table_03_individual_asr_outputs.md', ['Sample', 'Reference', 'Sarvam', 'Whisper Tiny', 'Whisper Base'], table_3_rows)

# TABLE 4: COCO Dataset
write_md_table('table_04_coco_statistics.md', ['Statistic', 'Value'], [[k, f"{v:,}"] for k, v in coco_stats.items()])

# TABLE 5: RefCOCO Dataset
write_md_table('table_05_refcoco_statistics.md', ['Statistic', 'Value'], [[k, f"{v:,}"] for k, v in refcoco_stats.items()])

# TABLE 6: RefCOCO Linguistic
write_md_table('table_06_refcoco_linguistic.md', ['Metric', 'Value'], [[k, f"{v:,}" if isinstance(v, int) else str(v)] for k, v in refcoco_linguistic.items()])

# TABLE 7: Term Frequencies
t7_rows = []
t7_rows.append(['**Spatial Terms**', ''])
for k, v in refcoco_spatial.items(): t7_rows.append([k, f"{v:,}"])
t7_rows.append(['**Color Terms**', ''])
for k, v in refcoco_color.items(): t7_rows.append([k, f"{v:,}"])
t7_rows.append(['**Size Terms**', ''])
for k, v in refcoco_size.items(): t7_rows.append([k, f"{v:,}"])
write_md_table('table_07_term_frequencies.md', ['Term', 'Frequency'], t7_rows)

# TABLE 8: Contributions
write_md_table('table_08_contributions.md', 
    ['Stage 1 (Implemented)', 'Stage 2+ (Future Work)'],
    [
        ['Speech-command understanding pipeline', 'Visual grounding'],
        ['ASR comparison framework', 'Bounding-box prediction'],
        ['Semantic evaluation', 'Multimodal fusion'],
        ['COCO/RefCOCO preparation', 'Spatial reasoning'],
        ['Dataset integrity validation', 'Navigation'],
        ['Linguistic EDA', 'Real-time camera experiments'],
        ['Reproducible evaluation scripts/tests', 'Blind-user evaluation']
    ]
)

print("Tables generated successfully.")
