import cv2
import numpy as np
import time

def mark_brightest_spot(frame):
    # Convert to grayscale to find the brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find the pixel with the maximum intensity
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    # Mark the brightest spot
    cv2.circle(frame, max_loc, 10, (0, 255, 255), 2) # Mark with a yellow circle
    return frame

def mark_reddest_spot(frame):
    # DISCLAIMER! I got help from Elsa for this part, mine was not working correctly, since I did not use the green and the blue channels

    # Define the "reddest" as the pixel with the highest red intensity
    red_channel = frame[:, :, 2].astype(int)
    green_channel = frame[:, :, 1].astype(int)
    blue_channel = frame[:, :, 0].astype(int)
 
    red_diff = red_channel - ((green_channel + blue_channel) // 2)
    reddest_spot = np.unravel_index(np.argmax(red_diff), red_diff.shape)

    cv2.circle(frame, (reddest_spot[1], reddest_spot[0]), 10, (0, 0, 255), 2)
    return frame

def mark_bright_manual_loop(frame):
    # Manually find the brightest spot using a double for-loop

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Initialize variables for manual detection
    max_val = 0
    max_loc = (0, 0)
    # Iterate, using a double for-loop
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i, j] > max_val:
                max_val = gray[i, j]
                max_loc = (j, i)
    # Mark the brightest spot
    cv2.circle(frame, max_loc, 15, (255, 0, 0), 2) # Mark with a blue circle
    return frame

# Set up
# For the computer camera:
cap = cv2.VideoCapture(0)

# For the phone camera:
#cap = cv2.VideoCapture(1)

fps_measurements = []

while(True):
    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Mark the brightest spot using OpenCV
    start_built_in = time.time()
    frame = mark_brightest_spot(frame)
    built_in_time = time.time() - start_built_in

    # Mark the reddest spot
    frame = mark_reddest_spot(frame)

    # Find and mark the brightest spot using double for-loop
    start_manual = time.time()
    frame = mark_bright_manual_loop(frame)
    manual_time = time.time() - start_manual

    #print(f"Built-in function time: {built_in_time:.6f} seconds")
    #print(f"Manual loop time: {manual_time:.6f} seconds")

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_measurements.append(fps)
    # Add FPS text
    fps_text = f"FPS: {fps: .2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Add frame processing time text
    frame_time_text = f"Frame time: {1 / fps:.4} seconds"
    cv2.putText(frame, frame_time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Real-Time detection', frame)

    # Break the loop on a 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # For question three (c), comment out imshow and waitkey and measure for 200 frames...
    #if len(fps_measurements) > 200:
    #    break

# Release the capture and close OpenCV windows
cap.release()
#cv2.destroyAllWindows()

# Print statistics for analysis
print("FPS Stats: ")
print(f"Average FPS: {np.mean(fps_measurements): .2f}")