import os
import csv
import string
import matplotlib.pyplot as plt
from jiwer import wer
import sys

# Load .env first
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

from faster_whisper import WhisperModel

# Make sure we can import from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from speech.sarvam_stt import SarvamTranscriber

def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove punctuation, normalize whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def calculate_accuracy(expected: str, actual: str) -> float:
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    
    if not expected_norm and not actual_norm:
        return 100.0
    if not expected_norm:
        return 0.0
        
    error_rate = wer(expected_norm, actual_norm)
    accuracy = (1.0 - error_rate) * 100.0
    return max(0.0, min(100.0, accuracy))

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    references_path = os.path.join(base_dir, "references.csv")
    results_path = os.path.join(base_dir, "results.csv")
    graph_path = os.path.join(base_dir, "comparison_graph.png")
    audio_dir = os.path.join(base_dir, "audio")

    # Read references
    references = []
    with open(references_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            references.append(row)

    if not references:
        print("Error: references.csv is empty. Please provide English translations.")
        return

    # Initialize models
    print("Loading models...")
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    sarvam_transcriber = SarvamTranscriber()

    results = []
    whisper_accuracies = []
    sarvam_accuracies = []

    print(f"\nProcessing {len(references)} samples...\n")
    
    for i, ref in enumerate(references):
        audio_filename = ref["audio_file"]
        expected_english = ref["expected_english"]
        audio_path = os.path.join(audio_dir, audio_filename)
        
        print(f"Processing sample {i+1}/{len(references)}...")
        print(f"Sample: {audio_filename}")
        print(f"Expected:\n{expected_english}")
        
        # Whisper Tiny
        segments, _ = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            task="translate",
            language=None,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )
        whisper_output = " ".join(seg.text.strip() for seg in segments).strip()
        
        # Sarvam AI
        try:
            sarvam_res = sarvam_transcriber.transcribe(audio_path)
            sarvam_output = sarvam_res.get("text", "")
        except Exception as e:
            print(f"Sarvam AI Error: {e}")
            sarvam_output = ""
        
        # Calculate WER and Accuracy
        expected_norm = normalize_text(expected_english)
        whisper_norm = normalize_text(whisper_output)
        sarvam_norm = normalize_text(sarvam_output)
        
        whisper_wer = wer(expected_norm, whisper_norm) if expected_norm and whisper_norm else (0.0 if not expected_norm and not whisper_norm else 1.0)
        sarvam_wer = wer(expected_norm, sarvam_norm) if expected_norm and sarvam_norm else (0.0 if not expected_norm and not sarvam_norm else 1.0)
        
        whisper_acc = calculate_accuracy(expected_english, whisper_output)
        sarvam_acc = calculate_accuracy(expected_english, sarvam_output)
        
        whisper_accuracies.append(whisper_acc)
        sarvam_accuracies.append(sarvam_acc)
        
        print(f"\nWhisper Tiny:\n{whisper_output}")
        print(f"Whisper Accuracy:\n{whisper_acc:.2f}%")
        
        print(f"\nSarvam AI:\n{sarvam_output}")
        print(f"Sarvam Accuracy:\n{sarvam_acc:.2f}%\n")
        print("-" * 40)
        
        results.append({
            "audio_file": audio_filename,
            "expected_english": expected_english,
            "whisper_output": whisper_output,
            "sarvam_output": sarvam_output,
            "whisper_wer": f"{whisper_wer:.4f}",
            "sarvam_wer": f"{sarvam_wer:.4f}",
            "whisper_accuracy": f"{whisper_acc:.2f}",
            "sarvam_accuracy": f"{sarvam_acc:.2f}"
        })
        
    # Write results.csv
    with open(results_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "audio_file", "expected_english", "whisper_output", "sarvam_output",
            "whisper_wer", "sarvam_wer", "whisper_accuracy", "sarvam_accuracy"
        ])
        writer.writeheader()
        writer.writerows(results)
        
    # Calculate averages
    avg_whisper = sum(whisper_accuracies) / len(whisper_accuracies) if whisper_accuracies else 0
    avg_sarvam = sum(sarvam_accuracies) / len(sarvam_accuracies) if sarvam_accuracies else 0
    
    print("\n========================================")
    print("FINAL COMPARISON")
    print("========================================")
    print(f"Whisper Tiny Average Accuracy: {avg_whisper:.2f}%")
    print(f"Sarvam AI Average Accuracy: {avg_sarvam:.2f}%")
    
    # Generate graph
    labels = ["Whisper Tiny", "Sarvam AI"]
    values = [avg_whisper, avg_sarvam]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=["#4da6ff", "#ff9933"])
    plt.ylim(0, 100)
    plt.ylabel("Average Accuracy (%)")
    plt.title("Whisper Tiny vs Sarvam AI")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f"{height:.2f}%", ha='center', va='bottom', fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig(graph_path, dpi=100)
    print(f"\nGraph saved to:\nevaluation/comparison_graph.png")

if __name__ == "__main__":
    main()
