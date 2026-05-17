from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import os
import time

app = Flask(__name__)
# Updated CORS for better compatibility with Edge/Chrome
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the model once
model = YOLO("yolov8m.pt") 

@app.route('/upload', methods=['POST'])
def analyze_traffic():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    video_file = request.files['file']
    # Use a unique name or add a tiny delay to ensure OS handles the file
    video_path = "temp_video.mp4" 
    video_file.save(video_path)
    
    # Wait a split second for the file to be "released" by the OS
    time.sleep(0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video file"}), 500
    
    max_counts = {
        "car": 0, "bus": 0, "truck": 0, 
        "person": 0, "motorcycle": 0
    }

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 30th frame
        if frame_count % 30 == 0:
            results = model(frame, verbose=False)
            
            current_frame = {
                "car": 0, "bus": 0, "truck": 0, 
                "person": 0, "motorcycle": 0
            }
            
            for r in results:
                for box in r.boxes:
                    label = model.names[int(box.cls[0])]
                    if label in current_frame:
                        current_frame[label] += 1
            
            for key in max_counts:
                if current_frame[key] > max_counts[key]:
                    max_counts[key] = current_frame[key]
        
        frame_count += 1

    cap.release()

    # Important: Delete file AFTER releasing the capture
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except:
            pass # Ignore if file is busy

    total_vehicles = sum(max_counts.values())

    # Final JSON Response
    return jsonify({
        "total": total_vehicles,
        "car": max_counts["car"],
        "bus": max_counts["bus"],
        "truck": max_counts["truck"],
        "pedestrian": max_counts["person"],
        "motorcycle": max_counts["motorcycle"],
        "density": "High" if total_vehicles > 15 else "Low"
    })

if __name__ == '__main__':
    # Running with debug=False can sometimes help with Port conflicts on HP
    app.run(port=5000, debug=False)