
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import os

SENTENCES = [
    "Where is my black bottle?",
    "Where is my phone?",
    "What is the color of this bag?",
    "Is there a laptop in front of me?",
    "Please identify this object.",
    "Where is the red bag?",
    "Can you find my bottle?",
    "Where is the blue book?",
    "Is my phone on the table?",
    "Where is the laptop?"
]

def record_audio(filename, duration=5, samplerate=16000):
    print("3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("SPEAK!")
    
    # Record audio
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    
    # Calculate RMS (energy) to check for silence
    rms = np.sqrt(np.mean(recording**2))
    
    # 0.005 is a typical threshold for detecting near-silence in normalized float32 audio
    if rms < 0.005:  
        print("\nNo speech detected. Please record this sample again.\n")
        return False
        
    sf.write(filename, recording, samplerate)
    return True

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_dir, "audio")
    
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        
    print(f"Starting recording session for 10 samples.")
    print(f"Audio will be saved to: {audio_dir}\n")
    
    for i, sentence in enumerate(SENTENCES):
        filename = f"sample_{i+1:02d}.wav"
        filepath = os.path.join(audio_dir, filename)
        
        while True:
            print("-" * 50)
            print(f"Sample {i+1}/10")
            print(f"Please read the following sentence aloud:")
            print(f"\n--->  {sentence}  <---\n")
            
            input("Press ENTER when you are ready to start...")
            
            success = record_audio(filepath)
            if success:
                print(f"Sample {i+1} recorded successfully.")
                break
                
    print("\n" + "=" * 50)
    print("10 audio samples recorded successfully.")
    print(f"Exact location of the files: {audio_dir}")
    print("==================================================")

if __name__ == "__main__":
    main()
