from ultralytics import YOLO
import cv2

# โหลดโมเดล YOLO11
model = YOLO("yolo11n.pt")
video_path = r"C:\Users\nutpu\Downloads\1900-151662242_tiny.mp4"

class_list = model.names

cap = cv2.VideoCapture(video_path)

ratio = 1
line_position = 220
class_count = {}
crossed_ids  = set()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        ("Done !")
        break
    width = int(frame.shape[1]*ratio)
    hright = int(frame.shape[0]*ratio)
    
    
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 1)
    
    results = model.track(frame, persist=True, classes=[2], device="cpu", verbose=False) # ถ้ามี GPU ใช้ cuda
    if results[0].boxes.data is not None: # tensor ของ bounding boxes
        # เก็บค่าที่ต้องการใช้
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()
    for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2 
        class_name = class_list[class_idx]
        
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)
        
        if  cy > line_position and track_id not in crossed_ids:
            crossed_ids.add(track_id)
            class_count[class_name] = class_count.get(class_name, 0) + 1
    
    cv2.putText(frame, "66010097 khanisorn rimpear Car Count:", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    for class_name, count in class_count.items():
        cv2.putText(frame, f"{count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
  
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit (q)")
        break

cap.release()
cv2.destroyAllWindows()
