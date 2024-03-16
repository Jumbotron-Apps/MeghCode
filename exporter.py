from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.export(format="coreml", nms=True, task="track", tracker="bytetrack.yaml")