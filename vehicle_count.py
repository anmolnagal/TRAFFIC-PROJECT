import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

line_y = 250
count = 0
tracked_centers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # HIGH RESOLUTION (important 🔥)
    frame = cv2.resize(frame, (960, 720))

    # 🔥 IMPROVED DETECTION SETTINGS
    results = model(frame, conf=0.25, iou=0.45)[0]

    new_centers = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # vehicle classes
        if cls in [2, 3, 5, 7]:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            new_centers.append((cx, cy))

            # draw
            label = f"{conf:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

            # COUNT LOGIC
            for (px, py) in tracked_centers:
                if abs(cx - px) < 60:
                    if py < line_y and cy >= line_y:
                        count += 1

    tracked_centers = new_centers

    # draw line
    cv2.line(frame, (0, line_y), (960, line_y), (255,0,0), 3)

    # display count
    cv2.putText(frame, f"Count: {count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Vehicle Counter (Improved)", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

line_y = 250

# Counts
car = bike = bus = truck = 0
up = down = 0

prev = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))

    results = model(frame, conf=0.25)[0]

    curr = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        if cls not in [2,3,5,7]:
            continue

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        curr.append((cx, cy, cls))

        # draw
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # 🔥 MATCH + COUNT
        for (px, py, pcls) in prev:
            if abs(cx-px) < 40 and abs(cy-py) < 40:

                # DOWN
                if py < line_y and cy >= line_y:
                    down += 1
                    if cls == 2: car += 1
                    elif cls == 3: bike += 1
                    elif cls == 5: bus += 1
                    elif cls == 7: truck += 1

                # UP
                elif py > line_y and cy <= line_y:
                    up += 1

    prev = curr

    # line
    cv2.line(frame,(0,line_y),(800,line_y),(255,0,0),3)

    # display
    cv2.putText(frame,f"Car:{car}",(10,30),0,0.7,(0,0,255),2)
    cv2.putText(frame,f"Bike:{bike}",(10,60),0,0.7,(0,0,255),2)
    cv2.putText(frame,f"Bus:{bus}",(10,90),0,0.7,(0,0,255),2)
    cv2.putText(frame,f"Truck:{truck}",(10,120),0,0.7,(0,0,255),2)

    cv2.putText(frame,f"Up:{up}",(600,30),0,0.7,(255,0,0),2)
    cv2.putText(frame,f"Down:{down}",(600,60),0,0.7,(255,0,0),2)

    cv2.imshow("Traffic AI", frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()