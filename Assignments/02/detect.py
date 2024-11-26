import cv2
import numpy as np
import time
from sklearn.linear_model import RANSACRegressor

# Capture the video frame
cap = cv2.VideoCapture(0) # Use '0' for computer camera, and '1' for phone camera

# Apply edge detection, using the Canny edge detector
def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (3, 3), 1.0) # Tried (gray, (3, 3), 1.0) for smaller kernel size. Smooths the image to reduce noise while preserving edges. (had tried (5, 5), 1.5)
    edges = cv2.Canny(blurred, 100, 200) # Might need to adjust the thresholds, tried (100, 200), was (50, 150). lower thresholds detect weaker edges but can include noise. Higher thresholds focuse on strong edges
    return edges

# Extract edge coordinates, by converting edge-detected image into a list of coordinates
def get_edge_coordinates(edges, sampling_rate=1):
    coords = np.column_stack(np.where(edges > 0))
    return coords[::sampling_rate] # Use every k-th point

# Fit a line with RANSAC, identify the most prominent line among the edge points
def fit_line_ransac(coords):
    if len(coords) < 2:
        return None
    x = coords[:,1].reshape(-1, 1) # X-coordinates
    y = coords[:, 0] # Y-coordinates
    model = RANSACRegressor()
    model.fit(x, y)
    line_params = model.estimator_.coef_[0], model.estimator_.intercept_
    return line_params

# Draw the line on the video frame
def draw_line(frame, line_params):
    if line_params is None:
        return frame
    slope, intercept = line_params
    h, w, _ = frame.shape
    x_start, x_end = 0, w
    y_start, y_end = int(intercept), int(slope * w + intercept)
    return cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

# Combine all the steps to process the video in real time
while cap.isOpened():
    # Start time
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Downscale the frame, reduces the number of pixels to process. With fewer pixels, each frame gets processed quicker, resulting in higher FPS. However, if scaled down too much, we can lose fine details.
    small_frame = cv2.resize(frame, (640, 480)) # Resize to a smaller resolution (tied 320, 240 as well, was first 640, 480) (I think 320, 240 is too small)

    # Process and detect
    edges = detect_edges(small_frame)
    coords = get_edge_coordinates(edges, sampling_rate=10) # Tried to use all points first, but reduced to 10-th point for RANSAC to operate faster
    line_params = fit_line_ransac(coords)
    result_frame = draw_line(frame, line_params)

    # Calculate time
    elapsed_time = time.time() - start_time

    # Display elapsed time and fps on the frame
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(result_frame, f"Processing time: {elapsed_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (5, 5, 5), 2)

    # Show the result
    cv2.imshow("Line detection", result_frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()