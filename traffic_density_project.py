import cv2

# =========================
# TRAFFIC DENSITY FUNCTION
# =========================
def get_traffic_density(vehicle_count):
    if vehicle_count <= 5:
        return "LOW", (0, 255, 0)      # Green
    elif vehicle_count <= 15:
        return "MEDIUM", (0, 255, 255) # Yellow
    else:
        return "HIGH", (0, 0, 255)     # Red


# =========================
# VIDEO SOURCE
# =========================
VIDEO_PATH = 0
# 👉 Use 0 for webcam
# 👉 OR use file path like:
# VIDEO_PATH = r"C:\Users\Dell\Desktop\TRAFFIC-PROJECT\demo\traffic.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

# Check if video opens
if not cap.isOpened():
    print("❌ Error: Cannot open video source")
    exit()

print("✅ Video started successfully")

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠️ Frame not received / video ended")
        break

    frame = cv2.resize(frame, (900, 500))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Background subtraction
    fgmask = fgbg.apply(gray)

    # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # filter small noise
        if area > 800:
            vehicle_count += 1

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # =========================
    # DENSITY CALCULATION
    # =========================
    density, color = get_traffic_density(vehicle_count)

    # =========================
    # DISPLAY INFO
    # =========================
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Traffic: {density}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Visual bar
    cv2.rectangle(frame, (30, 100),
                  (30 + vehicle_count * 15, 120),
                  color, -1)

    # =========================
    # SHOW OUTPUT
    # =========================
    cv2.imshow("Traffic Density System", frame)
    cv2.imshow("Mask", fgmask)

    # Press ESC to exit
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()