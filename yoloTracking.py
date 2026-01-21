from ultralytics import YOLO
import cv2

# Load a pretrained YOLO26 pose model (e.g., 'yolo26n-pose.pt' for the nano model)
# The model file will be downloaded automatically on the first run
model = YOLO("yolo26m-pose.pt")

# Set the source to the default webcam (usually 0).
# The 'show=True' argument automatically displays the results in a pop-up window
# and 'stream=True' makes it memory efficient for video processing.
results = model.predict(source="0", show=True, stream=True)

# Iterate through the results if you need to access specific data points (optional)
# The `show=True` argument already handles the display, but this loop
# allows for additional processing if needed.
for result in results:
    # You can access keypoints, bounding boxes, etc. here
    keypoints = result.keypoints
    # For example, print the number of people detected in the frame
    # print(f"Detected {len(keypoints)} people") 
    pass

# Note: The script will keep running and displaying the live feed
# until you press 'q' in the display window or stop the script.
