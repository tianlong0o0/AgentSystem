from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("models/yolo11n.pt")

# Run inference on an image
results = model("1.png")  # list of 1 Results object
new_detected_classes = [result.names[int(box.cls[0])] for result in results for box in result.boxes]
results[0].show()