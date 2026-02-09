import cv2
import numpy as np
from scipy.spatial import distance as dist
import os


def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def approximate_ear_from_bbox(eye_bbox):
    """Approximate EAR from bounding box"""
    x, y, w, h = eye_bbox
    aspect_ratio = w / (h + 1e-6)
    ear = 0.35 - (aspect_ratio * 0.05)
    return max(0.05, min(0.4, ear))


def detect_eyes(frame, face_cascade, eye_cascade):
    """Detect eyes using OpenCV Haar Cascades"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    eyes = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        detected_eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in detected_eyes:
            abs_x = x + ex
            abs_y = y + ey
            eyes.append((abs_x, abs_y, ew, eh))
    
    return eyes


def debug_adaptive_blink(video_path, min_blinks=2, max_frames=300, frame_interval=3, 
                         show_window=False, save_frames=True):
    """
    ADAPTIVE Debug Blink Detection with visualization
    
    Parameters:
    - video_path: path ke video
    - min_blinks: minimum blinks to find (default: 2)
    - max_frames: max frame yang diproses (default: 300)
    - frame_interval: interval frame (default: 3)
    - show_window: show cv2 window (default: False)
    - save_frames: save debug frames (default: True)
    """
    
    print(f"\n🔍 ADAPTIVE DEBUG BLINK DETECTION")
    print(f"Video: {video_path}")
    print(f"Looking for at least {min_blinks} blinks\n")
    
    # Load cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eye_cascade_tree = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video Info:")
    print(f"   Total frames: {total_frames_in_video}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Duration: {total_frames_in_video/fps:.1f}s\n")
    
    # Constants
    EAR_THRESHOLD = 0.20
    CONSEC_FRAMES = 2
    
    blink_count = 0
    ear_history = []
    frame_count = 0
    processed_frames = 0
    missing_faces = 0
    counter = 0
    
    faces_detected_count = 0
    eyes_detected_count = 0
    
    # Create output directory
    if save_frames:
        debug_dir = "debug_frames_adaptive"
        os.makedirs(debug_dir, exist_ok=True)
        print(f"💾 Saving frames to: {debug_dir}/\n")
    
    saved_frame_count = 0
    
    print("🎬 Processing frames...")
    print("="*60)
    
    while cap.isOpened() and processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % frame_interval != 0:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        debug_frame = frame.copy()
        frame_info = []
        
        detected_this_frame = False
        
        if len(faces) > 0:
            faces_detected_count += 1
            
            for (x, y, w, h) in faces:
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                frame_info.append(f"Face: {w}x{h}")
                
                roi_gray = gray[y:y+h, x:x+w]
                
                # Try both cascades
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) == 0:
                    eyes = eye_cascade_tree.detectMultiScale(roi_gray)
                    frame_info.append("Tree cascade used")
                
                if len(eyes) >= 2:
                    eyes_detected_count += 1
                    detected_this_frame = True
                    
                    # Draw eyes
                    for idx, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                        abs_x = x + ex
                        abs_y = y + ey
                        cv2.rectangle(debug_frame, (abs_x, abs_y), 
                                    (abs_x+ew, abs_y+eh), (0, 255, 0), 2)
                        cv2.putText(debug_frame, f"Eye{idx+1}", (abs_x, abs_y-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Calculate EAR
                    leftEAR = approximate_ear_from_bbox(eyes[0])
                    rightEAR = approximate_ear_from_bbox(eyes[1])
                    ear = (leftEAR + rightEAR) / 2.0
                    ear_history.append(ear)
                    
                    frame_info.append(f"Eyes: {len(eyes)}")
                    frame_info.append(f"EAR: {ear:.3f}")
                    
                    # Blink detection
                    if ear < EAR_THRESHOLD:
                        counter += 1
                        frame_info.append(f"⚠️ Below threshold! ({counter})")
                        color = (0, 0, 255)
                    else:
                        if counter >= CONSEC_FRAMES:
                            blink_count += 1
                            frame_info.append(f"👁️ BLINK #{blink_count}!")
                            print(f"   👁️ Blink #{blink_count} at frame {frame_count} (EAR was {ear:.3f})")
                        counter = 0
                        color = (0, 255, 0)
                    
                    # EAR text
                    cv2.putText(debug_frame, f"EAR: {ear:.3f}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    frame_info.append(f"Eyes: {len(eyes)} (need 2)")
                    missing_faces += 1
        else:
            frame_info.append("No face")
            missing_faces += 1
        
        # Add text overlay
        y_offset = 30
        for info in frame_info:
            cv2.putText(debug_frame, info, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Bottom info
        cv2.putText(debug_frame, f"Frame: {frame_count}/{total_frames_in_video}", 
                   (10, debug_frame.shape[0]-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Blinks: {blink_count}", 
                   (10, debug_frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save important frames
        if save_frames:
            # Save: first 5, every 20th, and frames with blinks
            if saved_frame_count < 5 or frame_count % 60 == 0 or counter > 0:
                output_path = os.path.join(debug_dir, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(output_path, debug_frame)
                saved_frame_count += 1
        
        # Show window
        if show_window:
            cv2.imshow('Adaptive Debug', debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        processed_frames += 1
        
        # Early stop if found enough blinks
        if blink_count >= min_blinks and processed_frames >= 50:
            print(f"\n✅ Found {blink_count} blinks, stopping early at frame {frame_count}")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # DETAILED SUMMARY
    print("\n" + "="*60)
    print("📊 DETAILED DEBUG SUMMARY")
    print("="*60)
    
    print(f"\n📹 Processing Stats:")
    print(f"   Total video frames: {total_frames_in_video}")
    print(f"   Frames checked: {frame_count}")
    print(f"   Frames processed: {processed_frames}")
    print(f"   Frames with face: {faces_detected_count} ({faces_detected_count/max(processed_frames,1)*100:.1f}%)")
    print(f"   Frames with eyes: {eyes_detected_count} ({eyes_detected_count/max(processed_frames,1)*100:.1f}%)")
    print(f"   Missing detection: {missing_faces}")
    
    print(f"\n👁️  Blink Stats:")
    print(f"   Total blinks detected: {blink_count}")
    
    if len(ear_history) > 0:
        print(f"\n📈 EAR Statistics:")
        print(f"   Samples: {len(ear_history)}")
        print(f"   Min EAR: {min(ear_history):.3f}")
        print(f"   Max EAR: {max(ear_history):.3f}")
        print(f"   Avg EAR: {np.mean(ear_history):.3f}")
        print(f"   Std EAR: {np.std(ear_history):.3f}")
        print(f"   Current threshold: {EAR_THRESHOLD}")
        
        # EAR distribution
        below_threshold = sum(1 for ear in ear_history if ear < EAR_THRESHOLD)
        print(f"   Frames below threshold: {below_threshold}/{len(ear_history)} ({below_threshold/len(ear_history)*100:.1f}%)")
    
    print(f"\n🔍 DIAGNOSIS:")
    print("="*60)
    
    if faces_detected_count == 0:
        print("❌ CRITICAL: No faces detected!")
        print("   Solutions:")
        print("   • Check if video shows face clearly")
        print("   • Verify video file is not corrupted")
        print("   • Try different video angle")
        
    elif eyes_detected_count == 0:
        print("❌ CRITICAL: Face found but NO eyes detected!")
        print("   Possible causes:")
        print("   • Eyes obscured (sunglasses, hair, hand)")
        print("   • Side profile (need frontal face)")
        print("   • Very low resolution")
        print("   • Poor lighting")
        
    elif blink_count == 0:
        print("⚠️  WARNING: Eyes detected but NO blinks!")
        
        if len(ear_history) > 0:
            min_ear = min(ear_history)
            max_ear = max(ear_history)
            
            if min_ear > EAR_THRESHOLD:
                print(f"\n   🔴 ROOT CAUSE: EAR never drops below threshold!")
                print(f"   • Threshold: {EAR_THRESHOLD:.3f}")
                print(f"   • Min EAR: {min_ear:.3f} (never reached threshold)")
                print(f"\n   💡 SOLUTION: Lower the threshold!")
                suggested = min_ear - 0.02
                print(f"   • Try threshold: {suggested:.3f}")
                print(f"   • Edit blink_detector: EAR_THRESHOLD = {suggested:.2f}")
            
            elif max_ear - min_ear < 0.05:
                print(f"\n   🔴 ROOT CAUSE: EAR too stable (no variation)!")
                print(f"   • Range: {min_ear:.3f} - {max_ear:.3f} (diff: {max_ear-min_ear:.3f})")
                print(f"\n   💡 POSSIBLE CAUSES:")
                print(f"   • Person not blinking in video")
                print(f"   • Video too short")
                print(f"   • Fake video (deepfake)")
            
            else:
                print(f"\n   ℹ️  EAR looks normal but no blinks detected")
                print(f"   • This might actually indicate a deepfake!")
                print(f"   • Or person genuinely didn't blink")
        
        print(f"\n   📊 Check saved frames in '{debug_dir}/' to verify")
    
    else:
        print(f"✅ SUCCESS: Detected {blink_count} blinks")
        if blink_count < 2:
            print(f"   ⚠️  Low blink count - might indicate deepfake")
    
    if save_frames:
        print(f"\n📁 Debug frames saved: {debug_dir}/")
        print(f"   Saved {saved_frame_count} frames")
        print(f"   Check these to see what was detected!")
    
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("Enter video path: ")
    
    # Set show_window=True if you want to see live detection
    debug_adaptive_blink(
        video_path, 
        min_blinks=2,
        max_frames=300,
        frame_interval=2,
        show_window=True,  # Change to True to see window
        save_frames=True
    )