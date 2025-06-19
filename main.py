import os
import cv2
import time
import csv
from ultralytics import YOLO
from kcf import Tracker

def resize_frame(frame, width=640):
    h, w = frame.shape[:2]
    r = width / float(w)
    dim = (width, int(h * r))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def run_yolo_only(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes.data
        if len(detections) > 0:
            x1, y1, x2, y2 = map(int, detections[0][:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "YOLO Detection", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        frame_count += 1
        frame = resize_frame(frame)
        cv2.putText(frame, "YOLO Only", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("YOLO Only", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    cap.release()
    cv2.destroyAllWindows()
    total_time = end_time - start_time
    fps = frame_count / total_time
    return frame_count, total_time, fps

def run_yolo_kcf(video_path, model):
    cap = cv2.VideoCapture(video_path)
    tracker = Tracker()
    tracking = False
    frame_count = 0
    yolo_resets = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        use_yolo = False

        if not tracking:
            use_yolo = True
        else:
            try:
                (x, y, w, h), apce = tracker.update(frame)
                if w <= 0 or h <= 0 or apce < 0.02:
                    raise ValueError(f"Invalid or low APCE: {apce:.4f}")
                status = f"KCF Tracking | APCE: {apce:.4f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'APCE: {apce:.4f}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print(f"[YOLO Reset] Frame {frame_count}: {e}")
                yolo_resets += 1
                use_yolo = True

        if use_yolo:
            results = model(frame)
            detections = results[0].boxes.data
            if len(detections) > 0:
                x1, y1, x2, y2 = map(int, detections[0][:4])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker.init(frame, bbox)
                tracking = True
                status = "YOLO Detection (Reinit)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "YOLO Detection", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                tracking = False
                status = "YOLO Detection (No Target)"

        frame = resize_frame(frame)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("YOLO + KCF", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    cap.release()
    cv2.destroyAllWindows()
    total_time = end_time - start_time
    fps = frame_count / total_time
    return frame_count, total_time, fps, yolo_resets

def process_folder(folder_path, model):
    results_file = os.path.join(folder_path, "results.csv")
    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "YOLO_Frames", "YOLO_Time", "YOLO_FPS",
                         "Hybrid_Frames", "Hybrid_Time", "Hybrid_FPS", "YOLO_Resets"])

        for file in os.listdir(folder_path):
            if file.endswith(".mp4"):
                video_path = os.path.join(folder_path, file)
                print(f"\n▶ Processing: {file}")

                y_frames, y_time, y_fps = run_yolo_only(video_path, model)
                h_frames, h_time, h_fps, yolo_resets = run_yolo_kcf(video_path, model)

                writer.writerow([file, y_frames, f"{y_time:.2f}", f"{y_fps:.2f}",
                                 h_frames, f"{h_time:.2f}", f"{h_fps:.2f}", yolo_resets])

                print(f"✓ Done: {file}")

def main():
    folder_path = "C:\\Users\\adams\\Desktop\\crazyflie-videos\\"
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return

    model = YOLO("crazy_yolo.pt")
    process_folder(folder_path, model)
    print("\n✅ All videos processed. Results saved in 'results.csv'.")

if __name__ == "__main__":
    main()
