
try:
    import google.genai
    print("google.genai: OK")
except ImportError:
    print("google.genai: MISSING")

try:
    from object_tracker import ObjectTracker
    print("ObjectTracker: OK")
    tracker = ObjectTracker(None) # Should print warning if sam2 missing
    print(f"Tracker Available: {tracker.available}")
except ImportError as e:
    print(f"ObjectTracker Import Failed: {e}")
except Exception as e:
    print(f"ObjectTracker Init Failed: {e}")
