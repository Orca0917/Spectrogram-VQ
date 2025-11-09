import os
import numpy as np
import librosa
from tqdm import tqdm


SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0
FMAX = 8000


def wav_to_mel(wav_path):
    wav, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    wav, _ = librosa.effects.trim(wav)
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0
    )
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return mel.astype(np.float32)


def build_mel_dataset(wav_dir, mel_dir):
    os.makedirs(mel_dir, exist_ok=True)
    wav_files = [f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")]
    
    if not wav_files:
        print("No wav files found!")
        return
    
    print(f"Processing {len(wav_files)} files")
    success = 0
    
    for wav_name in tqdm(wav_files):
        try:
            mel = wav_to_mel(os.path.join(wav_dir, wav_name))
            np.save(os.path.join(mel_dir, f"{os.path.splitext(wav_name)[0]}.npy"), mel)
            success += 1
        except:
            continue
    
    print(f"Success: {success}/{len(wav_files)}")


if __name__ == "__main__":
    build_mel_dataset(
        wav_dir="/workspace/data/wavs",
        mel_dir="/workspace/tts/DCTTS/data/mels"
    )
