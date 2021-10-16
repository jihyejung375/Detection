import cv2

cap = cv2.VideoCapture("highway.mp4")

# Object detection from Stable Camera
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    height,width, _ = frame.shape

    # Extract Region of interest
    roi = frame[340:720, 500:800]

    # Object Detection
    mask = object_detector.apply(roi)
    contours, _= cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(roi,[cnt],-1, (0,255,255),2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask",mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()