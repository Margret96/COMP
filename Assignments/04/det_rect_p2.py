import cv2
import numpy as np
import time

# Connect to the camera 
cap = cv2.VideoCapture(0) # Use 1 for the phone camera (The output is a bit strange with it though)

if not cap.isOpened():
    print("Error: Cannot access the camera")
    exit()

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

def find_intersections(lines):
    # Find intersections of lines using cross product (Homogeneous coordinates)
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = np.cross([lines[i][0], lines[i][1], 1], [lines[i][2], lines[i][3], 1])
            line2 = np.cross([lines[j][0], lines[j][1], 1], [lines[j][2], lines[j][3], 1])
            intersection = np.cross(line1, line2)
            if intersection[2] != 0: # Convert from homogeneous to Cartesian coordinates
                x = int(intersection[0] / intersection[2])
                y = int(intersection[1] / intersection[2])
                intersections.append((x, y))
    return intersections

def warp_perspective(frame, intersections):
    # Warp the perspective using intersections
    if len(intersections) < 4:
        return None
    
    intersections = np.array(intersections[:4], dtype="float32")
    rect = order_points(intersections)

    # Calculate the aspect ratio
    width_a = np.sqrt(((rect[2][0] - rect[3][0])**2) + ((rect[2][1] - rect[3][1])**2))
    width_b = np.sqrt(((rect[1][0] - rect[0][0])**2) + ((rect[1][1] - rect[0][1])**2))
    max_width = int(max(width_a, width_b))

    height_a = np.sqrt(((rect[1][0] - rect[2][0])**2) + ((rect[1][1] - rect[2][1])**2))
    height_b = np.sqrt(((rect[0][0] - rect[3][0])**2) + ((rect[0][1] - rect[3][1])**2))
    max_height = int(max(height_a, height_b))

    dst = np.array([[0, 0], [max_width -1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (max_width, max_height))
    return warped

def order_points(pts):
    # Order points to top-left, top-right, bottom-right and bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)] # Top-left
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[2] = pts[np.argmax(s)] # Bottom-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left

    return rect

# Initialise variables for FPS measurement (for video rate)
prev_time = 0
fps = 0 # Current FPS

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # FPS calculations
    current_time = time.time()
    fps = 1 / (current_time - prev_time) # Calculate FPS
    prev_time = current_time

     # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Smooth the image to reduce noise, maybe not necessary

    # Apply Canny edge detection
    # Determine thresholds by using median
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v)) # Lower threshold
    upper = int(min(255, 1.33 * v)) # Upper threshold
    edges = cv2.Canny(blurred, lower, upper) # Use if I need to reduce noise!!!! (or just use 50, 150)
    #edges = cv2.Canny(gray, 50, 150)
    # Parameters that can be adjusted and what they do:
    # Threshold1 and Threshold2: Low and high hysteresis thresholds
    # Tried range for Threshold1 = 50-100, and Threshold2 = 150-200

    # Apply Hough line transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=150, maxLineGap=50)
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

        # Find intersection
        intersections = find_intersections(prom_lines)

        # Draw intersections on the frame
        valid_intersections = []
        for (x, y) in intersections:
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                valid_intersections.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) # Draw green dots for intersections
        
        # Warp perspective if valid intersections are found
        if len(valid_intersections) >= 4:
            warped = warp_perspective(frame, valid_intersections)
            if warped is not None:
                cv2.imshow("Warped image", warped)

    # Display results
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Edges", edges) # Display the edge-detected image
    cv2.imshow("Detected lines", frame) # Display the lines overlaid on the original frame

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()