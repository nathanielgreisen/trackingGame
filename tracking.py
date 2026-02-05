import cv2
import mediapipe as mp
import time
import numpy as np

# --------------------------------------------- #
#   Hand and body tracker, written in python
#  for CBU1 Robotics by Nathaniel Greisen. It
#  uses MediaPipe and OpenCV to process images.
# --------------------------------------------- #

#### CONFIG ####
NUM_PEOPLE = 2 # The maximum number of people that can be detected
DETECTION_MIN = 0.5 # The minimum confidence score for the pose/hand detection to be considered successful
PRESENCE_MIN = 0.5 # The minimum confidence score of pose/hand presence score in the pose landmark detection.
TRACKING_MIN = 0.5 # The minimum confidence score for the pose/hand tracking to be considered successful.

# Set to True to keep enabled, False to disable
ENABLE_HAND = True 
ENABLE_POSE = True
ENABLE_FACE = True
ENABLE_VISUALIZATION = True

hand_model_path = "models/hand_landmarker.task" # path to hand model
pose_model_path = "models/pose_landmarker_lite.task" # path to pose model
face_model_path = "models/face_landmarker.task" # path to face model

### INITIALIZATIONS ###

# Create options and base functions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

# Hand Options
handOptions = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2*NUM_PEOPLE,
    min_hand_detection_confidence=DETECTION_MIN,
    min_hand_presence_confidence=PRESENCE_MIN,
    min_tracking_confidence=TRACKING_MIN,
)

# Pose Options
poseOptions = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=NUM_PEOPLE,
    min_pose_detection_confidence=DETECTION_MIN,
    min_pose_presence_confidence=PRESENCE_MIN,
    min_tracking_confidence=TRACKING_MIN,
)

# Face Options
faceOptions = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=NUM_PEOPLE,
    min_face_detection_confidence=DETECTION_MIN,
    min_face_presence_confidence=PRESENCE_MIN,
    min_tracking_confidence=TRACKING_MIN,
)

# Create landmarkers
handLandmarker = HandLandmarker.create_from_options(handOptions)
poseLandmarker = PoseLandmarker.create_from_options(poseOptions)
FaceLandmarker = FaceLandmarker.create_from_options(faceOptions)

# Get camera
cap = cv2.VideoCapture(0)

# Hand and Pose presets
HAND_LINE_GROUPS = np.array([
              [0,1,2,3,4], # Thumb
              [0,5,6,7,8], # Index finger
              [0,9,10,11,12], # Middle Finger
              [0,13,14,15,16], # Ring Finger
              [0,17,18,19,20], # Pinky
              [1,5,9,13,17] # Connect fingers
], dtype=object)

POSE_LINE_GROUPS = np.array([
            (11, 13), (13, 15), # left arm
            (12, 14), (14, 16), # right arm
            (11, 12), # shoulders
            (23, 24), # hips
            (11, 23), (12, 24), # torso
            (23, 25), (25, 27), # left leg
            (24, 26), (26, 28), # right leg
            ], dtype=np.int32)

FACE_LINE_GROUPS = np.array([
            [0,1,2,3,4] # temp
], dtype=object)

HAND_MAIN_COLOR = (0,255,0)
HAND_ACCENT_COLOR = (0,0,255)
POSE_MAIN_COLOR = (255,0,0)
POSE_ACCENT_COLOR = (0,0,255)
FACE_MAIN_COLOR = (128,128,0)
FACE_ACCENT_COLOR = (0,128,128)

prev_time = time.time()

### CODE START ###

def drawLandmarks(landmarks, frame, h, w, main_color, accent_color, line_groups, type):
    if not landmarks:
        return frame
    
    for lm_list in landmarks:
        # Position
        positions = np.array([(int(lm.x*w), int(lm.y * h)) for lm in lm_list])
        
        
        # Draw all circles
        for i, (cx, cy) in enumerate(positions):
            cv2.circle(frame, (cx, cy), 4, main_color, -1)
            cv2.putText(frame, str(i), (cx+5, cy-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, main_color, 1)
        
        # Draw pose lines
        if type == "pose":
            for start_idx, end_idx in line_groups:
                cv2.line(frame, tuple(positions[start_idx]),
                         tuple(positions[end_idx]), accent_color, 1)
        # Draw hand lines
        elif type == "hand" or "":
            for section in line_groups:
                for j in range(len(section) - 1):
                    start_pt = tuple(positions[section[j]])
                    end_pt = tuple(positions[section[j + 1]])
                    cv2.line(frame, start_pt, end_pt, accent_color, 1)
    return frame

def main():
    # Start FPS calculations
    frame_count = 0
    fps_update_time = time.time()
    fps = 0.0
    
    while True:
        
        ret, frame = cap.read() # init camera
        if not ret:
            break

        h, w = frame.shape[:2]
        
        frame = cv2.flip(frame, 1)


        # Prepare image for mediapipe processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - prev_time) * 1000)

        # Hand logic
        if ENABLE_HAND:
            handResult = handLandmarker.detect_for_video(mp_image, timestamp_ms)
            if ENABLE_VISUALIZATION:
                frame = drawLandmarks(handResult.hand_landmarks, frame, h, w, 
                              HAND_MAIN_COLOR, HAND_ACCENT_COLOR,
                              HAND_LINE_GROUPS, "hand")
        if ENABLE_POSE:
            poseResult = poseLandmarker.detect_for_video(mp_image, timestamp_ms)
            if ENABLE_VISUALIZATION: 
                frame = drawLandmarks(poseResult.pose_landmarks, frame, h, w, 
                              POSE_MAIN_COLOR, POSE_ACCENT_COLOR,
                              POSE_LINE_GROUPS, "pose")
        if ENABLE_FACE:
            faceResult = FaceLandmarker.detect_for_video(mp_image, timestamp_ms)
            if ENABLE_VISUALIZATION:
                frame = drawLandmarks(faceResult.face_landmarks, frame, h, w,
                              FACE_MAIN_COLOR, FACE_ACCENT_COLOR,
                              FACE_LINE_GROUPS, "face")

        # FPS calculation and display
        frame_count += 1
        current_time = time.time()
        if current_time - fps_update_time >= 1.0:  # Update FPS every second
            fps = frame_count / (current_time - fps_update_time)
            frame_count = 0
            fps_update_time = current_time
        
        # Display FPS on frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Display frames
        cv2.imshow("Body Tracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # esc to quit
            break
    
    # Close tracker
    cap.release()
    cv2.destroyAllWindows()
    handLandmarker.close()
    poseLandmarker.close()

def generateVisuals():
    
    frame = returnFrame()
    
    h, w = frame.shape[:2]

    # Prepare image for mediapipe processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - prev_time) * 1000)
    
    handPositions = None
    posePositions = None
    
    if ENABLE_HAND:
        handResult = handLandmarker.detect_for_video(mp_image, timestamp_ms)
        for lm_list in handResult.hand_landmarks:
            handPositions = np.array([(int(lm.x*w), int(lm.y * h)) for lm in lm_list])
        if ENABLE_VISUALIZATION:
            frame = drawLandmarks(handResult.hand_landmarks, frame, h, w, 
                              HAND_MAIN_COLOR, HAND_ACCENT_COLOR,
                              HAND_LINE_GROUPS, is_pose=False)
        
    if ENABLE_POSE:
        poseResult = poseLandmarker.detect_for_video(mp_image, timestamp_ms)
        for lm_list in poseResult.pose_landmarks:
            posePositions = np.array([(int(lm.x*w), int(lm.y * h)) for lm in lm_list])
            
        if ENABLE_VISUALIZATION:
            frame = drawLandmarks(poseResult.pose_landmarks, frame, h, w, 
                              POSE_MAIN_COLOR, POSE_ACCENT_COLOR,
                              POSE_LINE_GROUPS, is_pose=True)
            
    return frame, handPositions, posePositions

def returnFrame():
    ret, frame = cap.read() # init camera
    if not ret:
        return
        
    frame = cv2.flip(frame, 1)
    
    return frame

if __name__ == "__main__":
    main()