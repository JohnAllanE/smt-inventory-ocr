# LLM created
# Too slow for use on laptop (cpu only)

import easyocr
import cv2
import numpy as np # Used implicitly by OpenCV, good practice to include

# --- 1. INITIALIZE OCR READER (Only run once!) ---
print("Initializing EasyOCR reader. This may take a moment...")
# Set 'en' for English. Adjust 'gpu' based on your PyTorch installation.
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR reader initialized.")

# --- 2. INITIALIZE WEBCAM ---
# 0 refers to the default camera (try 1, 2, etc., if it doesn't work)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # 3. CAPTURE IMAGE
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    # Get frame dimensions
    height, width, _ = frame.shape
    
    # --- 4. REGION OF INTEREST (ROI) ISOLATION ---
    # Define a central area where you will hold the SMT bag label.
    # We'll use a 40% margin on all sides as a starting point.
    x_start = int(width * 0.2)
    x_end = int(width * 0.8)
    y_start = int(height * 0.2)
    y_end = int(height * 0.8)
    
    # Crop the frame to the ROI
    roi_frame = frame[y_start:y_end, x_start:x_end]
    
    # Draw a rectangle on the main frame to show the user where to hold the label
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    
    # --- 5. IMAGE PREPROCESSING (The Optimization Challenge) ---
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a simple threshold (binarization) to create stark black-on-white text
    # You will need to fine-tune the threshold value (e.g., 150)
    _, thresh_roi = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # OPTIONAL: Display the processed image for debugging
    # cv2.imshow('Processed ROI', thresh_roi)

    # --- 6. OCR EXECUTION ---
    # Use the processed (thresh_roi) image for best results
    results = reader.readtext(thresh_roi)
    
    # --- 7. TEXT POST-PROCESSING & DISPLAY ---
    # This is the step where you filter the OCR output (results)
    # and use regex to extract the Part Number.
    
    inventory_data = {}
    
    for (bbox, text, prob) in results:
        # Filter for text with reasonable confidence (e.g., > 50%)
        if prob > 0.5:
            # Clean up the text (remove non-alphanumeric, spaces, etc.)
            clean_text = text.strip()
            
            # Print for immediate feedback
            print(f"OCR Found: '{clean_text}' (Confidence: {prob:.2f})")
            
            # For now, just display the text on the main frame
            cv2.putText(frame, clean_text, (x_start, y_end + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Display the final frame
    cv2.imshow('SMT Inventory Scanner', frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 8. CLEANUP ---
cap.release()
cv2.destroyAllWindows()