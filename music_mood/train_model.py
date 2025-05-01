import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from extract_features import create_dataset, save_features, load_features

song_folder = "./music_mood/songs"

if os.path.exists("openl3_features.npy") and os.path.exists("openl3_labels.npy"):
    X, y = load_features("openl3")
else:
    X, y = create_dataset(song_folder)
    save_features(X, y, "openl3")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
ConfusionMatrixDisplay(cm, display_labels=['happy', 'sad', 'angry']).plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()  #open the image in a window


joblib.dump(model, "mood_classifier.pkl")