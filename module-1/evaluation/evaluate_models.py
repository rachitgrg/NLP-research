import os
import csv
import sys
import time
import soundfile as sf
import statistics

# Load .env first
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from speech.sarvam_stt import SarvamTranscriber
from evaluation.whisper_adapter import WhisperEvalAdapter
from evaluation.metrics import compute_wer, compute_cer, compute_exact_match, normalize_text

class ModelWrapper:
    def __init__(self, name: str, transcriber):
        self.name = name
        self.transcriber = transcriber

    def transcribe_with_latency(self, audio_path: str):
        start = time.perf_counter()
        try:
            if hasattr(self.transcriber, 'transcribe'):
                res = self.transcriber.transcribe(audio_path)
            else:
                res = self.transcriber(audio_path)
            text = res.get("text", "")
            error = None
        except Exception as e:
            text = ""
            error = str(e)
            print(f"[{self.name}] Error: {error}")
            
        end = time.perf_counter()
        latency = end - start
        
        return {
            "text": text,
            "latency": latency,
            "error": error
        }

def get_audio_duration(audio_path: str) -> float:
    try:
        with sf.SoundFile(audio_path) as f:
            return f.frames / f.samplerate
    except Exception:
        return 0.0

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    references_path = os.path.join(base_dir, "references.csv")
    results_path = os.path.join(base_dir, "results.csv")
    audio_dir = os.path.join(base_dir, "audio")

    # Read references
    references = []
    if os.path.exists(references_path):
        with open(references_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                references.append(row)

    if not references:
        print("Error: references.csv is empty or missing.")
        return

    # Initialize models
    print("Loading models...")
    models = [
        ModelWrapper("sarvam", SarvamTranscriber()),
        ModelWrapper("whisper_tiny", WhisperEvalAdapter(model_size="tiny")),
        ModelWrapper("whisper_base", WhisperEvalAdapter(model_size="base"))
    ]

    results = []
    
    # Track stats for each model
    stats = {m.name: {"wer": [], "cer": [], "em": [], "latency": []} for m in models}

    print(f"\nProcessing {len(references)} samples...\n")
    
    for i, ref in enumerate(references):
        sample_id = ref["audio_file"]
        reference = ref["expected_english"]
        audio_path = os.path.join(audio_dir, sample_id)
        
        duration = get_audio_duration(audio_path)
        
        print(f"Processing sample {i+1}/{len(references)}: {sample_id}")
        
        row = {
            "sample_id": sample_id,
            "reference": reference,
            "audio_duration_seconds": f"{duration:.2f}"
        }
        
        for m in models:
            out = m.transcribe_with_latency(audio_path)
            
            # Since the current WAV files are silent, they will produce empty transcripts.
            # normalize_text returns empty string.
            # jiwer.wer raises ValueError if reference is empty, but our reference is NOT empty.
            # However, if hypothesis is empty, jiwer.wer will return 1.0 (or >1).
            # Let's ensure compute_wer/cer handles empty hypothesis gracefully.
            
            # Actually, metrics.py uses jiwer directly. 
            text = out["text"]
            latency = out["latency"]
            
            try:
                wer = compute_wer(reference, text)
            except Exception:
                wer = 1.0
                
            try:
                cer = compute_cer(reference, text)
            except Exception:
                cer = 1.0
                
            try:
                em = compute_exact_match(reference, text)
            except Exception:
                em = False
                
            row[f"{m.name}_text"] = text
            row[f"{m.name}_wer"] = f"{wer:.4f}"
            row[f"{m.name}_cer"] = f"{cer:.4f}"
            row[f"{m.name}_exact_match"] = str(em)
            row[f"{m.name}_latency_seconds"] = f"{latency:.4f}"
            
            stats[m.name]["wer"].append(wer)
            stats[m.name]["cer"].append(cer)
            stats[m.name]["em"].append(1.0 if em else 0.0)
            stats[m.name]["latency"].append(latency)
            
        results.append(row)
        print("-" * 40)
        
    # Write results.csv
    fieldnames = ["sample_id", "reference", "audio_duration_seconds"]
    for m in models:
        fieldnames.extend([
            f"{m.name}_text",
            f"{m.name}_wer",
            f"{m.name}_cer",
            f"{m.name}_exact_match",
            f"{m.name}_latency_seconds"
        ])
        
    with open(results_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print("\n========================================")
    print("FINAL EVALUATION SUMMARY")
    print("========================================")
    for m in models:
        m_stats = stats[m.name]
        mean_wer = statistics.mean(m_stats["wer"])
        mean_cer = statistics.mean(m_stats["cer"])
        em_rate = statistics.mean(m_stats["em"]) * 100.0
        mean_latency = statistics.mean(m_stats["latency"])
        median_latency = statistics.median(m_stats["latency"])
        
        print(f"--- {m.name.upper()} ---")
        print(f"Mean WER:           {mean_wer:.4f}")
        print(f"Mean CER:           {mean_cer:.4f}")
        print(f"Exact Match Rate:   {em_rate:.2f}%")
        print(f"Mean Latency:       {mean_latency:.4f}s")
        print(f"Median Latency:     {median_latency:.4f}s\n")
        
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
