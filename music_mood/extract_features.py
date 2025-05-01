import os
import numpy as np
import soundfile as sf
import openl3

SONG_DIR = os.path.join(os.path.dirname(__file__), "songs")

def extract_audio_features(file_path):
    try:
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        emb, _ = openl3.get_audio_embedding(
            audio, sr, input_repr="mel256", content_type="music", embedding_size=512
        )
        return emb.mean(axis=0)
    except Exception as e:
        print(f"Failed to extract features from {file_path}: {e}")
        return np.zeros(512)

def save_features(X, y, prefix="openl3"):
    np.save(f"{prefix}_features.npy", X)
    np.save(f"{prefix}_labels.npy", y)

def load_features(prefix="openl3"):
    return np.load(f"{prefix}_features.npy"), np.load(f"{prefix}_labels.npy")

def create_dataset(song_folder):
    X, y = [], []
    emotions = {'happy': 0, 'sad': 1, 'angry': 2}

    for emotion, label in emotions.items():
        emotion_folder = os.path.join(song_folder, emotion)
        if not os.path.isdir(emotion_folder): continue
        for song_file in os.listdir(emotion_folder):
            if song_file.endswith('.mp3') or song_file.endswith('.wav'):
                try:
                    song_path = os.path.join(emotion_folder, song_file)
                    print(f"Processing: {song_path}")
                    audio_features = extract_audio_features(song_path)
                    X.append(audio_features)
                    y.append(label)
                except Exception as e:
                    print(f"Failed processing {song_file}: {e}")
    return np.array(X), np.array(y)