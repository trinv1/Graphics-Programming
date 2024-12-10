import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

#Load pretained YOLOv8n model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "vid1.mp4"
cap = cv2.VideoCapture(video_path)

