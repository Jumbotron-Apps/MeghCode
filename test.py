import cv2
import numpy as np

cap = cv2.VideoCapture('sample1.mp4')  # Video file directory must go here

if not cap.isOpened():
    print("Error: Couldn't open the video.")
    exit()

_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Lucas-Kanade
lk_params = dict(winSize=(40, 40),
                 maxLevel=7,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

# Mouse function


def select_point(event, x, y, flags, params):
    global point, point_selected, old_points, paused, rectangle_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)
        paused = False
    # elif event == cv2.EVENT_RBUTTONDOWN:
    #     rectangle_coords = (x, y, 50, 120)  # Initialize rectangle coordinates (adjust size as needed)
    #     cv2.rectangle(frame, (x, y), (x + 50, y + 120), (0, 255, 0), 2)
    #     region_inside_rectangle = frame[y:y+120, x:x+50]
    #     cv2.imwrite('region_inside_rectangle_image.jpg', region_inside_rectangle)


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)

point_selected = False
point = ()
old_points = None
rectangle_coords = None
paused = True

while True:
    if not paused:
        # Read a new frame
        ret, frame = cap.read()

        # Check if the frame is empty (end of the video)
        if not ret:
            print("End of video.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(
            gray_frame, (5, 5), 0)  # Apply GaussianBlur

        if point_selected is True:
            cv2.circle(frame, point, 5, (0, 0, 2575), )

            if old_points is not None:
                new_points, status, error = cv2.calcOpticalFlowPyrLK(
                    old_gray, gray_frame, old_points, None, **lk_params)

                if status is not None and all(status.flatten()):
                    old_points = new_points

                    x, y = new_points.ravel()
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        old_gray = gray_frame.copy()

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(17)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
