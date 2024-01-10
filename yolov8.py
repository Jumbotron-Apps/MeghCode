from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

video_path = 'sample1.mp4'

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
roi = cv2.selectROI(frame)
cv2.destroyAllWindows()

x, y, w, h = roi

ret = True

while ret:
    ret, frame = cap.read()

    roi_frame = frame[y:y+h, x:x+w]
    results = model.track(roi_frame, persist=True)

    for result in results.xyxy:
        result.xyxy[:, 0] += x
        result.xyxy[:, 1] += y
        frame = result.plot(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
