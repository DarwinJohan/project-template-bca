import cv2
import numpy as np
from collections import deque
import os

# ------------------------
# Setup debug folder
# ------------------------
DEBUG_DIR = "debug_frames"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ------------------------
# EAR approximation from bbox
# ------------------------
def approximate_ear_from_bbox(eye_bbox):
    x, y, w, h = eye_bbox
    ratio = h / (w + 1e-6)
    ear = max(0.1, min(0.35, ratio * 0.35))
    return ear

# ------------------------
# Detect eyes with Haar Cascade
# ------------------------
def detect_eyes(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        detected_eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        for (ex, ey, ew, eh) in detected_eyes:
            # Filter: mata biasanya di atas setengah wajah
            if ey + eh/2 > h/2:
                continue
            abs_x = x + ex
            abs_y = y + ey
            eyes.append((abs_x, abs_y, ew, eh))
    return eyes

# ------------------------
# Blink analysis
# ------------------------
def analyze_blink(video_path, min_blinks=2, max_frames=300, frame_interval=3, verbose=True):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    EAR_THRESHOLD = 0.20
    CONSEC_FRAMES = 2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return {"success": False, "reason": "cannot_open_video"}

    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    blink_count = 0
    counter = 0
    processed_frames = 0
    missing_faces = 0
    ear_history = deque(maxlen=5)  # smoothing

    if verbose:
        print(f"\n👁️ Blink analysis: {video_path} | Total frames: {total_frames_in_video}")

    frame_count = 0
    while cap.isOpened() and processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        eyes = detect_eyes(frame, face_cascade, eye_cascade)
        if len(eyes) < 2:
            missing_faces += 1
            processed_frames += 1
            continue

        # Ambil dua mata pertama
        leftEAR = approximate_ear_from_bbox(eyes[0])
        rightEAR = approximate_ear_from_bbox(eyes[1])
        ear = (leftEAR + rightEAR) / 2.0
        ear_history.append(ear)
        smooth_ear = np.mean(ear_history)

        # Blink detection
        blinked = False
        if smooth_ear < EAR_THRESHOLD:
            counter += 1
        else:
            if counter >= CONSEC_FRAMES:
                blink_count += 1
                blinked = True
                if verbose:
                    print(f"   👁️ Blink #{blink_count} at frame {frame_count}")
            counter = 0

        # ------------------------
        # Draw eyes and blink info for debug
        # ------------------------
        for (ex, ey, ew, eh) in eyes[:2]:
            color = (0, 0, 255) if blinked else (0, 255, 0)
            cv2.rectangle(frame, (int(ex), int(ey)), (int(ex+ew), int(ey+eh)), color, 2)
        cv2.putText(frame, f"Blink: {blink_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        debug_filename = os.path.join(DEBUG_DIR, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(debug_filename, frame)

        processed_frames += 1

        # Early stop
        if blink_count >= min_blinks and processed_frames >= 50:
            if verbose:
                print(f"   ✅ Found {blink_count} blinks, stopping early")
            break

    cap.release()

    if not ear_history:
        return {"success": False, "reason": "no_faces_detected"}

    total_frames_with_face = len(ear_history)
    missing_ratio = missing_faces / processed_frames if processed_frames > 0 else 0
    fps = 30
    duration_seconds = (processed_frames * frame_interval) / fps
    blink_rate_per_minute = (blink_count / duration_seconds) * 60 if duration_seconds > 0 else 0

    avg_ear = float(np.mean(ear_history))
    std_ear = float(np.std(ear_history))

    # Suspicious detection
    suspicious = False
    reasons = []
    if blink_rate_per_minute < 8:
        suspicious = True
        reasons.append("low_blink_rate")
    elif blink_rate_per_minute > 35:
        suspicious = True
        reasons.append("high_blink_rate")
    if std_ear < 0.015:
        suspicious = True
        reasons.append("low_ear_variance")
    if missing_ratio > 0.6:
        suspicious = True
        reasons.append("many_missing_faces")
    if avg_ear < 0.15 or avg_ear > 0.35:
        suspicious = True
        reasons.append("abnormal_ear")

    result = {
        "success": True,
        "total_frames": total_frames_with_face,
        "processed_frames": processed_frames,
        "missing_faces": missing_faces,
        "blink_count": blink_count,
        "blink_rate_per_minute": blink_rate_per_minute,
        "avg_ear": avg_ear,
        "std_ear": std_ear,
        "missing_ratio": missing_ratio,
        "suspicious": suspicious,
        "reasons": reasons
    }

    if verbose:
        print("\n📊 Blink Summary:")
        print(f"- Processed frames: {processed_frames}")
        print(f"- Total frames with face: {total_frames_with_face}")
        print(f"- Missing faces: {missing_faces}")
        print(f"- Total blinks: {blink_count}")
        print(f"- Blink rate: {blink_rate_per_minute:.1f} per minute")
        print(f"- Avg EAR: {avg_ear:.3f}")
        print(f"- Std EAR: {std_ear:.3f}")
        print("\n🧠 Interpretation:")
        if suspicious:
            print("⚠️ Suspicious:", ", ".join(reasons))
        else:
            print("✅ Looks normal")

    return result
