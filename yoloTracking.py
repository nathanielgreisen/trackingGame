from ultralytics import YOLO
import cv2

model = YOLO("yolo26n-pose.pt")



results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)

# results = model.predict(source="0", show=True, stream=True)

# for result in results:
#     keypoints = result.keypoints
#     pass
