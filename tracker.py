from ultralytics import YOLO
import cv2 as cv
import time
import torch
import logging
import numpy as np

# Initialize the logger
logging.basicConfig(level=logging.INFO)


def select_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def select_object(frame, model, tracker_config):
    processed_frame = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    results = model.track(frame, tracker=tracker_config,
                          persist=True, device=select_device())
    annotated_frame = results[0].plot()

    def click_event(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            boxes = results[0].boxes.cpu().numpy().xyxy
            distances = np.sqrt(
                (boxes[:, 0] - x) ** 2 + (boxes[:, 1] - y) ** 2)
            selected_idx = np.argmin(distances)
            selected_id = results[0].boxes.cpu(
            ).numpy().id[selected_idx].item()
            params['selected_id'] = selected_id
            params['clicked'] = True

    cv.imshow("Select Object", annotated_frame)
    cv.waitKey(1)
    click_params = {'clicked': False, 'selected_id': None}
    cv.setMouseCallback("Select Object", click_event, click_params)

    while not click_params['clicked']:
        cv.waitKey(1)

    cv.destroyWindow("Select Object")
    return click_params['selected_id']


def process_video(video_path, model, tracker_config, selected_id):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error opening video file")
        return

    frame_count = 0
    last_box = None

    while cap.isOpened():
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
        success, frame = cap.read()
        if not success:
            break

        start = time.perf_counter()

        if frame_count % 1 == 0:
            processed_frame = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            results = model.track(frame, tracker=tracker_config,
                                  persist=True, device=select_device())
            for result in results:
                boxes = result.boxes.cpu().numpy()
                if selected_id in boxes.id:
                    idx = boxes.id.tolist().index(selected_id)
                    last_box = (boxes[idx].xyxy[0] * 1).tolist()

        if last_box is not None:
            cv.rectangle(
                frame, (int(last_box[0]), int(last_box[1])), (int(last_box[2]), int(last_box[3])), (0, 255, 0), 2)

        end = time.perf_counter()
        fps = 1 / (end - start)
        # annotated_frame = results[0].plot()

        cv.putText(frame, f"FPS: {int(fps)}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow("Tracker", frame)
        
        frame_count += 1

    cap.release()
    cv.destroyAllWindows()

# Main Execution


model = YOLO('yolov8n.pt')
video_path = 'video1.mp4'
tracker_config = "bytetrack.yaml"

cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    logging.error("Error opening video file")
else:
    success, first_frame = cap.read()
    if success:
        selected_id = select_object(first_frame, model, tracker_config)
        logging.info(f"Selected Object ID: {selected_id}")
        process_video(video_path, model, tracker_config, selected_id)
        cap.release()
