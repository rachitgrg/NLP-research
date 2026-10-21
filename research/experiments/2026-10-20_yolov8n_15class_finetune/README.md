# EXP-03 · YOLOv8n fine-tune on 15 indoor classes (COCO train2014)

| | |
|---|---|
| Dates | Started 2026-10-14 23:45 (`train_batch0.jpg`); resumed from `last.pt` 2026-10-20 02:10 (`args.yaml`); finished 2026-10-20 06:14 (`results.csv`, `best.pt`) |
| Produced by | `module2_vision/training/train.py` (launched via `run_training.bat` → `python train.py --resume`); commit `3fb09d9` |
| Base model | `yolov8n.pt` (Ultralytics, COCO-pretrained, 80 classes) |
| Data | `data/processed/coco_processed.csv` (notebook 02 output, 604,906 boxes) → filtered to 15 classes → image-level 80/20 split, `RANDOM_SEED = 42`; val capped at `MAX_VAL_IMAGES = 3000`. 46,688 train / 3,000 val images, all from **COCO train2014**. |
| Classes | person, chair, dining table, couch, bed, bottle, cup, bowl, laptop, cell phone, book, tv, backpack, umbrella, suitcase (`dataset.yaml`) |
| Hyper-parameters (`args.yaml`) | epochs 30, batch 8, imgsz 640, SGD lr0 0.01, close_mosaic 10, AMP on, `seed: 0`, deterministic, device 0 |
| Hardware / env | RTX 3050 6 GB Laptop GPU; global Python 3.13 (ultralytics currently 8.4.177 there; the exact version at training time was not recorded) |
| Weights | `module2_vision/training/runs/train/weights/best.pt` (gitignored, 6.5 MB) — the app's default model |

## Results (epoch 30 = best epoch; `results.csv`)

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.567 | 0.467 | 0.481 | 0.331 |

Curves and confusion matrices are in `curves/`; prediction vs label examples are in `val_examples/`.

## Caveats (read before citing)
1. **Likely leakage, so mAP is optimistic.** The base `yolov8n.pt` was pretrained on COCO train2017, which contains almost all train2014 images. The 3,000 "val" images were therefore probably seen during pretraining (`PROJECT_AUDIT.md` §6, inference, high confidence). Clean numbers need COCO val2017 images (not downloaded; roadmap R2).
2. **No baseline on the same split.** Stock `yolov8n.pt` was not evaluated on these 15 classes and this split.
3. `results_summary_as_written.txt` says "Validation was not run or failed": the resume overwrote the summary. The metrics above come from `results.csv`.
4. Training was resumed once, so `args.yaml` shows `resume:` and absolute paths from the old `module-2/` layout.

## Conclusion
The fine-tuned model is the default for the app. Desk trials (EXP-05) favoured it over stock in face-occlusion scenes. Paper-grade accuracy claims need the val2017 evaluation first.
