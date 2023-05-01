import numpy as np
import cv2

class KalmanFilter:
    def __init__(self, dt, x_std, y_std, process_noise):
        self.dt = dt
        self.A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = process_noise * np.array(
            [[dt ** 3 / 3, dt ** 2 / 2, 0, 0], [dt ** 2 / 2, dt, 0, 0], [0, 0, dt ** 3 / 3, dt ** 2 / 2],
             [0, 0, dt ** 2 / 2, dt]])
        self.R = np.array([[x_std ** 2, 0], [0, y_std ** 2]])
        self.P = np.diag([1000, 1000, 1000, 1000])
        self.x = np.zeros((4, 1))
        self.first_detection = True

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def correct(self, z):
        if self.first_detection:
            self.x[:2] = z
            self.first_detection = False
        else:
            y = z.reshape((2, 1)) - np.dot(self.H, self.x)
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            self.x = self.x + np.dot(K, y)
            I = np.eye(4)
            self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
