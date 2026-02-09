from emotion_detector import analyze_emotion
from blink_detector import analyze_blink  # Pure OpenCV - no dependency conflicts!

VIDEO_PATH = "fake/2.mp4"


def main():
    print("\n" + "="*50)
    print("🔍 DEEPFAKE DETECTION ANALYSIS")
    print("="*50)
    
    # =====================
    # LEVEL 1: EMOTION
    # =====================
    
    emotion_result = analyze_emotion(VIDEO_PATH)
    
    if not emotion_result["success"]:
        print("❌ Emotion analysis failed:", emotion_result["reason"])
        return
    
    # =====================
    # LEVEL 2: BLINK
    # =====================
    
    blink_result = analyze_blink(VIDEO_PATH)
    
    if not blink_result["success"]:
        print("❌ Blink analysis failed:", blink_result.get("reason"))
        if "message" in blink_result:
            print("💡", blink_result["message"])
        return
    
    # =====================
    # FINAL VERDICT
    # =====================
    
    print("\n" + "="*50)
    print("🎯 FINAL VERDICT")
    print("="*50)
    
    print(f"\n📊 Emotion Detection:")
    print(f"   - Suspicious: {'⚠️ YES' if emotion_result['suspicious'] else '✅ NO'}")
    if emotion_result['suspicious']:
        print(f"   - Reasons: {', '.join(emotion_result['reasons'])}")
    
    print(f"\n👁️  Blink Detection:")
    print(f"   - Suspicious: {'⚠️ YES' if blink_result['suspicious'] else '✅ NO'}")
    if blink_result['suspicious']:
        print(f"   - Reasons: {', '.join(blink_result['reasons'])}")
    
    # Overall assessment
    suspicious_count = sum([
        emotion_result['suspicious'],
        blink_result['suspicious']
    ])
    
    print(f"\n{'='*50}")
    if suspicious_count >= 2:
        print("🚨 VERDICT: HIGH PROBABILITY OF DEEPFAKE")
    elif suspicious_count == 1:
        print("⚠️  VERDICT: POSSIBLY DEEPFAKE - NEEDS FURTHER REVIEW")
    else:
        print("✅ VERDICT: LIKELY AUTHENTIC VIDEO")
    print("="*50)
    
    # Detailed metrics
    print("\n📈 Detailed Metrics:")
    print(f"   Emotion diversity: {emotion_result['emotion_diversity']}")
    print(f"   Blink rate: {blink_result['blink_rate_per_minute']:.1f}/min (normal: 15-20)")
    print(f"   Emotion confidence: {emotion_result['avg_confidence']:.1f}%")
    print(f"   Eye aspect ratio: {blink_result['avg_ear']:.3f}")
    

if __name__ == "__main__":
    main()