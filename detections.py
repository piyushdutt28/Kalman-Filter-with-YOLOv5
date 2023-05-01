import torch
import numpy as np
import cv2
from PIL import Image

class ObjectDetector:
    def __init__(self, model_path, device='cuda'):
        self.model = torch.load(model_path, map_location=device).autoshape()
        self.device = device
        self.classes = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def detect(self, image):
        """
        Performs object detection on an input image.

        :param image: the input image
        :return: a list of detected objects, where each object is represented as a dictionary containing the following
        keys: 'class' (the class of the object), 'confidence' (the confidence score of the detection), 'bbox' (the
        bounding box of the object)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        else:
            image = Image.fromarray(image).convert('RGB')

        # Convert image to tensor
        image_tensor = torch.as_tensor(np.array(image) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Perform object detection
        results = self.model(image_tensor)

        #
