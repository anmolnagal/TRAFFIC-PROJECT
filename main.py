import cv2
from ultralytics import YOLO

vehicle_model = YOLO("yolov8n.pt")
rickshaw_model = YOLO("autorickshaw.pt")

VEHICLE_CLASSES = {1,2,3,5,7}  # bicycle, car, motorbike, bus, truck

def iou(boxA, boxB):
    # box = [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

cap = cv2.VideoCapture("videos/traffic1.mp4")
cv2.namedWindow("Traffic", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Traffic", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vehicle_results = vehicle_model(frame, conf=0.4, verbose=False)[0]
    rickshaw_results = rickshaw_model(frame, conf=0.4, verbose=False)[0]

    vehicle_boxes = []
    for box in vehicle_results.boxes:
        cls = int(box.cls[0])
        if cls in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = vehicle_model.names[cls]
            vehicle_boxes.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y1-25),(x1+120,y1),(0,255,0),-1)
            cv2.putText(frame,label,(x1+5,y1-7),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

    for box in rickshaw_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Check if overlaps any vehicle box >50%
        overlap = any(iou([x1,y1,x2,y2], vb) > 0.5 for vb in vehicle_boxes)
        if overlap:
            continue  # skip this rickshaw box
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,255),2)
        cv2.rectangle(frame,(x1,y1-25),(x1+140,y1),(0,200,255),-1)
        cv2.putText(frame,"rickshaw",(x1+5,y1-7),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

    cv2.imshow("Traffic", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
