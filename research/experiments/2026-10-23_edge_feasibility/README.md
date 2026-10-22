# EXP-10 · Edge feasibility: the pipeline without a GPU (roadmap R9, Phase 5)

**Status (2026-10-23): partial.** Detector speed on CPU and small-Whisper latency on CPU are done. Whisper `small`/`medium`/`large-v3` on CPU, peak memory, and the optional ONNX/INT8 export are still to do.

| | |
|---|---|
| Question | Can the offline pipeline run without the laptop GPU, as a first indication for a phone- or glasses-class device (the final product target, `decisions.md` D-16)? |
| Hardware | AMD Ryzen 5 7640HS (6 cores), 15 GB RAM, Windows 11. GPU reference: RTX 3050 6 GB Laptop |
| Detectors | `tools/openvocab_benchmark.py --models <best.pt> yoloe-26s-seg.pt --device cpu --runs 30 --imgsz {640,320}` (2026-10-23 05:06). Model-only latency on one fixed COCO image, median over 30 runs after 10 warm-up runs. Rows also appended to `module2_vision/tools/results/openvocab_benchmark_2026-10-23.csv`. |
| Speech | `evaluation/multilingual_benchmark.py --manifest en_smoke_manifest.csv --backends whisper:tiny+hint whisper:base+hint --device cpu` (05:07, commit `b1969dc`); CPU `int8`, beam 5 |
| Environment | root `.venv`: torch 2.11.0+cu128, ultralytics 8.4.172, faster-whisper 1.2.1, Python 3.13.1 |

## Results

### Detectors, model-only latency per frame (`detectors_cpu_vs_gpu_2026-10-23.csv`)
| Model | GPU 640 px | CPU 640 px | CPU 320 px |
|---|---|---|---|
| Closed-set `best.pt` (YOLOv8n, 15 classes) | 8.1–8.9 ms | 22.6 ms (44 FPS) | 10.4 ms (96 FPS) |
| Open-vocabulary YOLOE-26s (text prompt) | 16.9–20.1 ms | 73.6 ms (13.6 FPS) | 29.2 ms (34 FPS) |
| Both together (open-vocab targets): sum | ≈ 25–29 ms | ≈ 96 ms (≈ 10 FPS) | ≈ 40 ms (≈ 25 FPS) |

The GPU rows come from EXP-06 (03:18–03:19, same file). "Both together" is the sum of the two medians, not a measured hybrid run.

**Accuracy cost of 320 px (sanity image only):** bottle confidences drop from 0.42/0.32 to 0.26 (closed-set), and the open-vocab model's from 0.76/0.70/0.26 to 0.30. Two of three bottles are lost at the 0.25 threshold. Detection accuracy at 320 px has **not** been measured on desk scenes.

### Speech recognition on CPU (10 English clips, language given)
| Model | Size on disk | Object accuracy | Median latency | Load time |
|---|---|---|---|---|
| Whisper tiny | 75 MB | 10/10 | 0.20 s | 0.48 s |
| Whisper base | 142 MB | 10/10 | 0.37 s | 0.30 s |

Files: `multilingual_2026-10-23_050706.csv`, `…_summary.csv`, `….json`.

## Conclusion so far
Without a GPU the closed-set path is comfortably real time at 640 px (44 FPS). The open-vocabulary path runs at ≈ 10 FPS at 640 px, or ≈ 25 FPS at 320 px at a still-unmeasured accuracy cost. Small Whisper models answer in well under 0.5 s on CPU. This supports, but does not prove, that a phone/glasses port is feasible. A laptop CPU is much faster than a phone SoC, so the paper should say "CPU-only on a laptop", not "on a phone".

## Caveats
- Model-only timings on one image; camera capture, drawing and speech are excluded.
- English only and one speaker for ASR; multilingual accuracy belongs to EXP-08.
- The large Whisper models (the likely multilingual choice) haven't been timed on CPU yet. They need to be downloaded first.
