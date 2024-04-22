from ultralytics import YOLO
import cv2
import math
import os

# Initialize the YOLO model
model = YOLO('best.pt')

# Reading the classes
classnames = ['Fireman', 'Fire', 'Smoke']


def detect_objects(input_path):
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load image
        image = cv2.imread(input_path)

        # Perform object detection on the image
        results = model(image)

        # Getting bbox, confidence, and class names information to work with
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence >30 and 0 <= Class < len(classnames):  # Check if Class is within range
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if classnames[Class] != 'Fireman':  # Skip displaying frame if Fireman is detected
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cv2.putText(image, f'{classnames[Class]} {confidence}%', (x1 + 8, y1 + 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # Display the image with detections
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # Open video file
        cap = cv2.VideoCapture(input_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection on the frame
            results = model(frame)

            # Getting bbox, confidence, and class names information to work with
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    Class = int(box.cls[0])
                    if confidence > 50 and 0 <= Class < len(classnames):  # Check if Class is within range
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        if classnames[Class] != 'Fireman':  # Skip displaying frame if Fireman is detected
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            cv2.putText(frame, f'{classnames[Class]} {confidence}%', (x1 + 8, y1 + 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            # Display the frame with detections
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Unsupported file format!")


# Example usage
# file_path = '../assets/images/img.jpg'
file_path = '../assets/videos/fire2.mp4'

detect_objects(file_path)