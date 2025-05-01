import joblib
import numpy as np
from extract_features import extract_audio_features
import os

model = joblib.load("mood_classifier.pkl")
scaler = joblib.load("scaler.pkl")
label_map = {0: "happy", 1: "sad", 2: "angry"}

def predict_song_mood(file_path):
    audio_features = extract_audio_features(file_path)
    features_scaled = scaler.transform([audio_features])
    prediction = model.predict(features_scaled)[0]
    mood = label_map.get(prediction, "Unknown")
    print(f"\n🎧 Predicted Mood: {mood}\n")

if __name__ == "__main__":
    print("Please drag and drop your .mp3 or .wav file into the terminal and press ENTER:")
    file_path = input("→ ").strip().strip("'").strip('"')

    if os.path.isfile(file_path):
        predict_song_mood(file_path)
    else:
        print("❌ File not found. Please make sure the path is correct.")