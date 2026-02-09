import cv2
import numpy as np
from deepface import DeepFace
from collections import Counter


def analyze_emotion(video_path, max_frames=50, frame_interval=5, verbose=True):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    emotion_history = []
    confidence_history = []
    missing_faces = 0
    processed_frames = 0
    frame_count = 0

    if verbose:
        print(f"\n🎥 Emotion analysis: {video_path}\n")

    while cap.isOpened() and processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_interval != 0:
            continue

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv"
            )

            emotion = result[0]["dominant_emotion"]
            confidence = result[0]["emotion"][emotion]

            emotion_history.append(emotion)
            confidence_history.append(confidence)

        except:
            missing_faces += 1

        processed_frames += 1

    cap.release()

    # =====================
    # SUMMARY CALCULATION
    # =====================

    if len(emotion_history) == 0:
        return {
            "success": False,
            "reason": "no_faces_detected"
        }

    emotion_counts = Counter(emotion_history)
    dominant_emotion, dominant_count = emotion_counts.most_common(1)[0]

    diversity = len(emotion_counts)
    avg_confidence = float(np.mean(confidence_history))
    missing_ratio = missing_faces / processed_frames

    suspicious = False
    reasons = []

    if diversity <= 2 and dominant_count / len(emotion_history) > 0.75:
        suspicious = True
        reasons.append("monotonous_emotion")

    if avg_confidence > 80:
        suspicious = True
        reasons.append("high_confidence")

    if missing_ratio > 0.5:
        suspicious = True
        reasons.append("many_missing_faces")

    result = {
        "success": True,
        "total_faces": len(emotion_history),
        "missing_faces": missing_faces,
        "emotion_diversity": diversity,
        "emotion_frequency": dict(emotion_counts),
        "dominant_emotion": dominant_emotion,
        "avg_confidence": avg_confidence,
        "missing_ratio": missing_ratio,
        "suspicious": suspicious,
        "reasons": reasons
    }

    # =====================
    # OPTIONAL PRINT
    # =====================

    if verbose:
        print("📊 Emotion Summary:")
        print(f"- Total detected faces: {result['total_faces']}")
        print(f"- Missing faces: {result['missing_faces']}")
        print(f"- Emotion diversity: {result['emotion_diversity']}")
        print(f"- Emotion frequency: {result['emotion_frequency']}")
        print(f"- Dominant emotion: {result['dominant_emotion']}")
        print(f"- Avg confidence: {result['avg_confidence']:.2f}%")

        print("\n🧠 Interpretation:")
        if suspicious:
            print("⚠️ Suspicious:", ", ".join(reasons))
        else:
            print("✅ Looks normal")

    return result
