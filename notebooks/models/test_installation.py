"""
Quick test script untuk verify semua library terinstall dengan benar
"""

print("🧪 Testing installations...\n")

# Test 1: MediaPipe
try:
    import mediapipe as mp
    print("✅ MediaPipe:", mp.__version__)
except Exception as e:
    print("❌ MediaPipe error:", e)

# Test 2: OpenCV
try:
    import cv2
    print("✅ OpenCV:", cv2.__version__)
except Exception as e:
    print("❌ OpenCV error:", e)

# Test 3: NumPy
try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except Exception as e:
    print("❌ NumPy error:", e)

# Test 4: SciPy
try:
    import scipy
    print("✅ SciPy:", scipy.__version__)
except Exception as e:
    print("❌ SciPy error:", e)

# Test 5: DeepFace
try:
    from deepface import DeepFace
    print("✅ DeepFace: installed")
except Exception as e:
    print("❌ DeepFace error:", e)

print("\n" + "="*50)
print("📊 Summary:")
print("="*50)

# Quick functionality test
try:
    import mediapipe as mp
    import cv2
    import numpy as np
    
    # Test MediaPipe face mesh initialization
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Create dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = face_mesh.process(dummy_frame)
    
    print("✅ MediaPipe face mesh: Working!")
    print("✅ All systems ready!")
    print("\n🚀 You can now run: python main.py")
    
except Exception as e:
    print("⚠️  Functionality test failed:", e)
    print("💡 Try running the main script anyway - might still work!")