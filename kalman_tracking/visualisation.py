import cv2

def visualize(frame, detections, predictions, classes, colors):
    for i, row in detections.iterrows():
        label = classes[int(row['class'])]
        confidence = row['confidence']
        x, y, w, h = int(row['xmin']), int(row['ymin']), int(row['xmax'] - row['xmin']), int(row['ymax'] - row['ymin'])
        color = colors[int(row['class'])]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for cls, (x, y) in predictions:
        color = colors[classes.index(cls)]
        cv2.circle(frame, (x, y), 5, color, -1)
        cv2.putText(frame, cls, (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame
