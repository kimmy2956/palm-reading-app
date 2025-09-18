from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

# This is a conceptual function to detect a palm using Computer Vision.
# It's a very simple and basic heuristic. For a real-world application,
# you would need a more sophisticated model trained on a large dataset.
def detect_palm(image):
    try:
        # Convert image to HSV color space for better skin tone detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color range for skin tones (This is a generic range)
        lower_skin = np.array([0, 20, 70], dtype="uint8")
        upper_skin = np.array([20, 255, 255], dtype="uint8")

        # Create a mask for skin pixels
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Calculate the percentage of skin pixels in the image
        skin_pixels = np.sum(skin_mask > 0)
        total_pixels = image.shape[0] * image.shape[1]
        skin_percentage = skin_pixels / total_pixels

        # If more than 10% of the image is skin, assume it's a palm.
        # This is a very rough estimate and can be inaccurate.
        is_palm = skin_percentage > 0.1
        
        return is_palm, "Palm detected."
    except Exception as e:
        print(f"Error in image processing: {e}")
        return False, "Image processing failed."

# Serve the HTML file
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# API endpoint for palm analysis
@app.route('/api/analyze-palm', methods=['POST'])
def analyze_palm_api():
    if 'file' not in request.files:
        return jsonify({'is_palm': False, 'message': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'is_palm': False, 'message': 'No selected file.'}), 400

    if file:
        try:
            # Read the image file bytes and convert to a numpy array for OpenCV
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({'is_palm': False, 'message': 'Invalid image file.'}), 400

            is_palm, message = detect_palm(img)
            
            return jsonify({'is_palm': is_palm, 'message': message}), 200

        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'is_palm': False, 'message': 'An error occurred during analysis.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
