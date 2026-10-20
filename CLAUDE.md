# CLAUDE.md

Assistive object finder for visually impaired users: voice request → camera detection → spoken direction guidance.
Status and evidence: `PROJECT_AUDIT.md`. Planned work: `IMPLEMENTATION_ROADMAP.md`.

## Layout
- `module-1/` voice: Sarvam STT (translate to English) + spaCy rule parser. Entry: `main.py`.
- `module-2/` data + training: notebooks 01–03, `train.py` (15-class YOLOv8n fine-tune).
- `module-3/` vision + guidance. Entry points: `main.py` (flags `--target/--voice/--navigate`), `assistant.py` (voice-first).
- `research/` academic outputs (figures, tables, findings). Do not regenerate without asking.

## Environment
- Use the root `.venv` (Python 3.13, CUDA torch, voice + vision deps): `.\.venv\Scripts\python`.
- `module-1/venv` and `module-3/venv` are legacy per-module envs; keep them working, don't delete them.
- Secrets: `module-1/.env` (`SARVAM_API_KEY`). Never print, commit, or copy it.

## Tests
- `cd module-1; ..\.venv\Scripts\python -m pytest tests -q`
- `cd module-3; ..\.venv\Scripts\python -m pytest tests -q`
- Tests mock the camera, YOLO, and TTS. Hardware behaviour needs a manual run.

## Conventions
- Run scripts from inside their module folder; each module puts its own root on `sys.path`.
- Module 1 and Module 3 both have a `pipeline` package. Module 3 loads Module 1's `hearing_pipeline.py` by file path (`module-3/pipeline/m3_pipeline.py`); never `import pipeline.hearing_pipeline` from Module 3.
- Detection dict contract: `{"class_name","class_id","confidence","bbox":[x1,y1,x2,y2]}`. Don't change it without updating all consumers.
- Supported targets come from the loaded model (`detector.class_names`), not from the COCO-80 list.
- All tunables live in `module-3/config/config.py` and can be overridden with env vars.
- Match the existing style: header comment block per file, type hints, small focused classes.

## Safety rules
- Never delete or overwrite `module-2/datasets/`, `outputs/`, `yolo_dataset/`, `runs/`, `*.pt`, notebooks, or `research/` outputs.
- Ask before installing packages, downloading weights or datasets, or (re)training.
- Don't claim distance or navigation accuracy that hasn't been measured.
