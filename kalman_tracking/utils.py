import torch
import cv2
import numpy as np
from PIL import Image
from models.yolov5 import load_model

def load_video(path):
    return cv2.VideoCapture(path)

def load_model():
    return load_model('yolov5s', pretrained=True)

def detect_objects(model, frame, confidence_threshold=0.5):
    results = model(frame)
    detections = results.pandas().xyxy[0]
    detections = detections[detections['confidence'] >= confidence_threshold]
    return detections
