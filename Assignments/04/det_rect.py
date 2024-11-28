import cv2
import numpy as np
import time

# Connect to the camera
cap = cv2.VideoCapture(0) # Use 1 for the phone camera

if not cap.isOpened():
    print("Error: Cannot access the camera")
    exit()

def filter_lines(lines, img_shape):
    # Filter lines based on length and position
    height, width = img_shape[:2]
    filtered = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            # Keep lines that are long enough and within bounds
            if length > 0.3 * min(height, width): # Lenght threshold (30% smaller dimension)
                filtered.append((x1, y1, x2, y2))
    return filtered

def extend_lines(x1, y1, x2, y2, scale_factor=1.5):
    dx = x2 - x1
    dy = y2 - y1

    # Extend the line by scale_factor in both directions
    x1_ext = int(x1 - scale_factor * dx)
    y1_ext = int(y1 - scale_factor * dy)
    x2_ext = int(x2 + scale_factor * dx)
    y2_ext = int(y2 + scale_factor * dy)

    return x1_ext, y1_ext, x2_ext, y2_ext

def get_prom_lines(lines, top_n=4):
    # Select the most prominent lines in the image
    lengths = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            lengths.append((length, (x1, y1, x2, y2)))
    
    # Sort line sby length in descending order
    lengths = sorted(lengths, key=lambda x: x[0], reverse=True)

    # Return the top N lines
    extended_lines = [extend_lines(x1, y1, x2, y2) for _, (x1, y1, x2, y2) in lengths[:top_n]]
    return extended_lines

# Initialise variables for FPS measurement (for video rate)
prev_time = 0
fps = 0 # Current FPS

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # FPS calc
    current_time = time.time()
    fps = 1 / (current_time - prev_time) # Calculate FPS
    prev_time = current_time

    # Step 1: convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Smooth the image to reduce noise, maybe not necessary

    # Step 2: Apply Canny edge detection
    # Determine thresholds by using median
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v)) # Lower threshold
    upper = int(min(255, 1.33 * v)) # Upper threshold
    edges = cv2.Canny(blurred, lower, upper) # Use if I need to reduce noise!!!!
    #edges = cv2.Canny(gray, 50, 150)
    # Parameters that can be adjusted and what they do:
    # Threshold1 and Threshold2: Low and high hysteresis thresholds
    # Tried range for Threshold1 = 50-100, and Threshold2 = 150-200

    # Step 3: Apply Hough line transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=30)
    # Parameters that can be adjusted and what they do:
    # rho: Distance resolution in pixels. Lower values give finer resolution
    # theta: Angle resolution in radians. (Tried np.pi/180 and np.pi/360 for finer resolution.)
    # threshold: Mininum number of votes to consider a line. Increase to reduce noise.
    # minLineLength: minimum length of line
    # minLineGap: length between lines to be considered two lines instead of one

    # Draw the detected lines on the original frame
    if lines is not None:
        #filtered_lines = filter_lines(lines, frame.shape)
        prom_lines = get_prom_lines(lines, top_n=4) # Get the 4 most prominent lines (possible to increase if needed)
        for x1, y1, x2, y2 in prom_lines: # Can change into filtered lines if needed (get more lines)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Draw the line in red

    # Step 4: Display the results
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Edges", edges) # Display the edge-detected image
    cv2.imshow("Detected lines", frame) # Display the lines overlaid on the original frame

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()