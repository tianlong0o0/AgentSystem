import pytest
from ultralytics import YOLO

@pytest.fixture
def init_model():
    model = YOLO("models/yolo11x.pt")  # 使用 YOLOv11 Extra-Large 模型
    yield model
    del model

def test_yolo(init_model):
    results = init_model("test/dog.jpg")
    detected_classes = [result.names[int(box.cls[0])] for result in results for box in result.boxes]
    assert "dog" in detected_classes