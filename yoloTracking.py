from ultralytics import YOLO
import cv2

# def main():
#     model = YOLO("best.pt")
#     results = model.train(
#         data="hand-keypoints.yaml",
#         epochs=100,
#         imgsz=640
#     )

# if __name__ == "__main__":
#     main()

model = YOLO("models/last.pt")

results = model.predict(source="0", show=True, stream=True)

for result in results:
    keypoints = result.keypoints
    pass
