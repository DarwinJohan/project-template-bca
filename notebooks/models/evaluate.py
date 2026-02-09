import os
import json
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from emotion_detector import analyze_emotion
from blink_detector_adaptive import analyze_blink


def analyze_video(video_path, verbose=False):
    """
    Analyze single video and return results
    """
    results = {
        "video_path": video_path,
        "emotion": None,
        "blink": None,
        "prediction": None,
        "confidence": 0
    }
    
    try:
        # Emotion analysis
        emotion_result = analyze_emotion(video_path, verbose=verbose)
        if emotion_result["success"]:
            results["emotion"] = emotion_result
        
        # Blink analysis
        blink_result = analyze_blink(video_path, verbose=verbose)
        if blink_result["success"]:
            results["blink"] = blink_result
        
        # Make prediction
        if emotion_result["success"] and blink_result["success"]:
            suspicious_count = sum([
                emotion_result['suspicious'],
                blink_result['suspicious']
            ])
            
            if suspicious_count >= 2:
                results["prediction"] = "FAKE"
                results["confidence"] = 0.9
            elif suspicious_count == 1:
                results["prediction"] = "FAKE"
                results["confidence"] = 0.6
            else:
                results["prediction"] = "REAL"
                results["confidence"] = 0.8
        
    except Exception as e:
        results["error"] = str(e)
    
    return results


def print_video_details(filename, result, label):
    """
    Print detailed analysis for each video
    """
    prediction = result.get("prediction", "ERROR")
    status = "✅" if prediction == label else "❌"
    
    # Header
    print(f"\n{status} {filename}")
    print(f"   Label: {label}  |  Predicted: {prediction}")
    
    # Emotion details
    if result.get("emotion"):
        emotion = result["emotion"]
        print(f"\n   📊 EMOTION ANALYSIS:")
        print(f"      Status: {'⚠️ SUSPICIOUS' if emotion['suspicious'] else '✅ NORMAL'}")
        
        if emotion['suspicious']:
            print(f"      Reasons: {', '.join(emotion['reasons'])}")
        
        # Emotion frequency
        print(f"      Emotions detected:")
        for emo, count in emotion['emotion_frequency'].items():
            percentage = (count / emotion['total_faces'] * 100) if emotion['total_faces'] > 0 else 0
            print(f"         - {emo}: {count} ({percentage:.1f}%)")
        
        print(f"      Dominant: {emotion['dominant_emotion']}")
        print(f"      Avg Confidence: {emotion['avg_confidence']:.1f}%")
        print(f"      Emotion Diversity: {emotion['emotion_diversity']}")
    else:
        print(f"\n   📊 EMOTION ANALYSIS: ERROR")
    
    # Blink details
    if result.get("blink"):
        blink = result["blink"]
        print(f"\n   👁️  BLINK ANALYSIS:")
        print(f"      Status: {'⚠️ SUSPICIOUS' if blink['suspicious'] else '✅ NORMAL'}")
        
        if blink['suspicious']:
            print(f"      Reasons: {', '.join(blink['reasons'])}")
        
        print(f"      Total Blinks: {blink['blink_count']}")
        print(f"      Blink Rate: {blink['blink_rate_per_minute']:.1f}/min (normal: 15-20)")
        print(f"      Avg EAR: {blink['avg_ear']:.3f}")
        print(f"      EAR Variance: {blink['std_ear']:.3f}")
    else:
        print(f"\n   👁️  BLINK ANALYSIS: ERROR")
    
    print(f"   " + "-"*50)


def process_dataset(fake_dir="fake", real_dir="real", output_file="results.json"):
    """
    Process all videos in fake and real directories
    """
    print("\n" + "="*60)
    print("🔬 DATASET BATCH PROCESSING - DETAILED MODE")
    print("="*60)
    
    # Get all video files
    fake_videos = []
    real_videos = []
    
    # Supported video formats
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    
    # Collect fake videos
    if os.path.exists(fake_dir):
        for file in sorted(os.listdir(fake_dir)):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                fake_videos.append(os.path.join(fake_dir, file))
        print(f"📁 Found {len(fake_videos)} videos in '{fake_dir}/'")
    else:
        print(f"⚠️  Folder '{fake_dir}/' not found")
    
    # Collect real videos
    if os.path.exists(real_dir):
        for file in sorted(os.listdir(real_dir)):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                real_videos.append(os.path.join(real_dir, file))
        print(f"📁 Found {len(real_videos)} videos in '{real_dir}/'")
    else:
        print(f"⚠️  Folder '{real_dir}/' not found")
    
    total_videos = len(fake_videos) + len(real_videos)
    print(f"📊 Total videos to process: {total_videos}")
    
    if total_videos == 0:
        print("❌ No videos found. Exiting...")
        return
    
    # Process all videos
    all_results = []
    
    # FAKE VIDEOS
    print("\n" + "="*60)
    print("Batch: FAKE")
    print("="*60)
    
    for video_path in fake_videos:
        filename = os.path.basename(video_path)
        print(f"\n🎬 Processing: {filename}...")
        result = analyze_video(video_path, verbose=False)
        result["ground_truth"] = "FAKE"
        all_results.append(result)
        
        # Print detailed results
        print_video_details(filename, result, "FAKE")
    
    # REAL VIDEOS
    print("\n" + "="*60)
    print("Batch: REAL")
    print("="*60)
    
    for video_path in real_videos:
        filename = os.path.basename(video_path)
        print(f"\n🎬 Processing: {filename}...")
        result = analyze_video(video_path, verbose=False)
        result["ground_truth"] = "REAL"
        all_results.append(result)
        
        # Print detailed results
        print_video_details(filename, result, "REAL")
    
    # Calculate metrics
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    emotion_suspicious_count = 0
    blink_suspicious_count = 0
    both_suspicious_count = 0
    
    for result in all_results:
        if result["prediction"]:
            total += 1
            ground_truth = result["ground_truth"]
            prediction = result["prediction"]
            
            if ground_truth == prediction:
                correct += 1
            
            if ground_truth == "FAKE" and prediction == "FAKE":
                true_positives += 1
            elif ground_truth == "REAL" and prediction == "FAKE":
                false_positives += 1
            elif ground_truth == "REAL" and prediction == "REAL":
                true_negatives += 1
            elif ground_truth == "FAKE" and prediction == "REAL":
                false_negatives += 1
            
            # Count suspicious by type
            if result.get("emotion") and result["emotion"]["suspicious"]:
                emotion_suspicious_count += 1
            if result.get("blink") and result["blink"]["suspicious"]:
                blink_suspicious_count += 1
            if result.get("emotion") and result.get("blink"):
                if result["emotion"]["suspicious"] and result["blink"]["suspicious"]:
                    both_suspicious_count += 1
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    precision = (true_positives / (true_positives + false_positives) * 100) if (true_positives + false_positives) > 0 else 0
    recall = (true_positives / (true_positives + false_negatives) * 100) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    print(f"\n🎯 Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"\n📈 Metrics:")
    print(f"   Precision: {precision:.1f}%")
    print(f"   Recall:    {recall:.1f}%")
    print(f"   F1-Score:  {f1_score:.1f}%")
    
    print(f"\n🔍 Confusion Matrix:")
    print(f"   TP (Fake→Fake): {true_positives}")
    print(f"   TN (Real→Real): {true_negatives}")
    print(f"   FP (Real→Fake): {false_positives}")
    print(f"   FN (Fake→Real): {false_negatives}")
    
    print(f"\n🚨 Suspicious Analysis:")
    print(f"   Emotion suspicious: {emotion_suspicious_count}/{total}")
    print(f"   Blink suspicious:   {blink_suspicious_count}/{total}")
    print(f"   Both suspicious:    {both_suspicious_count}/{total}")
    
    # Save results to JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_videos": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": {
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            },
            "suspicious_counts": {
                "emotion": emotion_suspicious_count,
                "blink": blink_suspicious_count,
                "both": both_suspicious_count
            }
        },
        "results": all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    # Process dataset
    results = process_dataset(
        fake_dir="fake",
        real_dir="real",
        output_file="evaluation_results_detailed.json"
    )