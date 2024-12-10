
import cv2
from ultralytics import YOLO

#Load pretained YOLOv8n model
model = YOLO('yolov8n.pt')

#Running inference on source
results = model(source = 'vid1.mp4', show = True, conf = 0.4)