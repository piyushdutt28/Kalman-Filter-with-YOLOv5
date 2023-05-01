import cv2
import numpy as np
from kalman import KalmanFilter

class Tracker:
    def __init__(self, classes, colors, dt=1/30.0, Q=1e-3 * np.eye(4), R=np.array([[100, 0], [0, 100]])):
        self.classes = classes
        self.colors = colors
        self.kfs = {}
        for cls in classes:
            self.kfs[cls] = KalmanFilter(dt=dt, Q=Q, R=R)

    def predict(self):
        for cls in self.classes:
            self.kfs[cls].predict()

    def update(self, detections):
        for i, row in detections.iterrows():
            label = self.classes[int(row['class'])]
            x, y, w, h = int(row['xmin']), int(row['ymin']), int(row['xmax'] - row['xmin']), int(row['ymax'] - row['ymin'])
            z = np.array([[x + w/2], [y + h/2]])
            self.kfs[label].update(z)

    def get_predictions(self):
        predictions = []
        for cls in self.classes:
            prediction = self.kfs[cls].get_prediction()
            if prediction is not None:
                x, y = prediction.flatten()
                predictions.append((cls, (int(x), int(y))))
        return predictions

    def draw_predictions(self, frame):
        for cls, (x, y) in self.get_predictions():
            color = self.colors[self.classes.index(cls)]
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.putText(frame, cls, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
