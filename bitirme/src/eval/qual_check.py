"""Veri kalite kontrol scripti: transcription boş mu, unicode bozuk mu vb."""
import json

def quality_report(coco_json_path):
    # TODO: Roboflow COCO alanlarına göre gerçek kontroller
    return {
        "empty_transcriptions": 0,
        "invalid_unicode": 0,
        "bbox_issues": 0
    }
