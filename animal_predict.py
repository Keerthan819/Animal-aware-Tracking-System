import os
import cv2
import numpy as np
import imageio
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# ==============================
# 🧩 PATH SETTINGS
# ==============================
# 👉 Change this to your actual input video path
input_video  = r"C:\Users\Dell\animal_tracking_dataset\raw_videos\cows\VID20251012183155.mp4"

# 👉 Output video will be created here
output_video = r"C:\Users\Dell\animal_tracking_dataset\output_videos\cowww_predict_output.mp4"
os.makedirs(os.path.dirname(output_video), exist_ok=True)

# ==============================
# 🧠 LOAD YOLO MODEL
# ==============================
# Make sure yolov8n.pt is downloaded and accessible from this path
# Either put yolov8n.pt in the same folder and use "yolov8n.pt"
# or give full path like: r"C:\Users\Dell\AnimalTracker\yolov8n.pt"
model = YOLO("yolov8n.pt")   # lightweight YOLO model

# ==============================
# ⚙️ VIDEO SETUP
# ==============================
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Could not open input video: {input_video}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width, height = int(cap.get(3)), int(cap.get(4))
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎞 Video info: {width}x{height} @ {fps:.2f} FPS ({frame_total} frames)")

# imageio writer (uses ffmpeg under the hood)
writer = imageio.get_writer(
    output_video,
    fps=fps,
    codec="libx264",
    format="ffmpeg",
    pixelformat="yuv420p",
    macro_block_size=1
)

# ==============================
# 📈 TRAJECTORY STORAGE
# ==============================
tracks_history = {}  # {track_id: [(x, y), ...]}  -> past trajectory
filters = {}         # {track_id: KalmanFilter}   -> motion model


# ==============================
# 🔬 KALMAN FILTER (Constant Acceleration Model)
# State: [x, y, vx, vy, ax, ay]^T
# ==============================
def create_kalman(dt=1.0):
    kf = KalmanFilter(dim_x=6, dim_z=2)

    # State transition matrix with constant acceleration
    kf.F = np.array([
        [1, 0, dt, 0, 0.5*dt*dt, 0],
        [0, 1, 0, dt, 0, 0.5*dt*dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    # We only observe position (x, y)
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ])

    # Covariance matrices
    kf.P *= 25.0     # Initial state covariance
    kf.R *= 4.0      # Measurement noise
    kf.Q = np.eye(6) * 0.5  # Process noise (smoothness)

    # Initial state vector
    kf.x = np.zeros((6, 1))
    return kf


# ==============================
# 🚀 MAIN LOOP (YOLO + ByteTrack + Trajectory)
# ==============================
frame_idx = 0

# Approximate dt from fps
dt = 1.0 / fps if fps > 0 else 1.0

for result in model.track(
    source=input_video,
    tracker="bytetrack.yaml",  # built-in tracker
    conf=0.4,
    iou=0.5,
    imgsz=640,
    persist=True,
    stream=True,
    show=False,
    save=False,
    # COCO "animal-like" classes: person(0) excluded, typical animals:
    # 15=cat,16=dog,17=horse,18=sheep,19=cow,20=elephant,21=bear,22=zebra,23=giraffe
    classes=[15, 16, 17, 18, 19, 20, 21, 22, 23]
):
    frame = result.plot()  # YOLO-drawn boxes, IDs, etc.
    frame_idx += 1

    # We'll draw extra stuff (past + future trajectories) on top of YOLO visual
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id

        if ids is not None:
            ids = ids.int().cpu().numpy()

            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = box[:4]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # --- Initialize Kalman filter for new track ID ---
                if obj_id not in filters:
                    kf = create_kalman(dt=dt)
                    kf.x[0, 0] = cx  # x position
                    kf.x[1, 0] = cy  # y position
                    filters[obj_id] = kf
                    tracks_history[obj_id] = []

                # --- Update Kalman filter with detection ---
                kf = filters[obj_id]
                kf.predict()
                kf.update([cx, cy])

                # Current filtered state (better than raw YOLO center)
                pred_x, pred_y = int(kf.x[0]), int(kf.x[1])

                # Store past trajectory
                tracks_history[obj_id].append((pred_x, pred_y))

                # ======================
                # 🟩 DRAW PAST TRAJECTORY
                # ======================
                past_pts = np.array(tracks_history[obj_id], dtype=np.int32)
                if len(past_pts) > 1:
                    for j in range(1, len(past_pts)):
                        cv2.line(frame,
                                 tuple(past_pts[j - 1]),
                                 tuple(past_pts[j]),
                                 (0, 255, 0), 2)  # green line

                # ======================
                # 🔴 FUTURE TRAJECTORY (Constant Acceleration)
                # ======================
                future_state = kf.x.copy()
                F_ca = kf.F.copy()

                future_points = []
                pred_steps = 30  # number of future points (~1 sec at 30 FPS)

                for _ in range(pred_steps):
                    future_state = F_ca @ future_state
                    fx, fy = int(future_state[0]), int(future_state[1])
                    future_points.append((fx, fy))

                # Draw future trajectory (red dots)
                for pt in future_points:
                    fx, fy = pt
                    # Only draw if within frame bounds
                    if 0 <= fx < frame.shape[1] and 0 <= fy < frame.shape[0]:
                        cv2.circle(frame, (fx, fy), 3, (0, 0, 255), -1)

                # Direction arrow based on near future
                if len(future_points) > 5:
                    start_pt = future_points[0]
                    end_pt = future_points[5]
                    cv2.arrowedLine(
                        frame,
                        start_pt,
                        end_pt,
                        (0, 0, 255),
                        2,
                        tipLength=0.4
                    )

                # Label with ID
                cv2.putText(
                    frame,
                    f"ID {obj_id}",
                    (pred_x + 10, pred_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

    # Write modified frame
    writer.append_data(np.asarray(frame))

    if frame_idx % 25 == 0:
        print(f"✅ Processed {frame_idx}/{frame_total} frames...")

writer.close()
cap.release()
print(f"\n✅ Tracking + Improved Trajectory prediction complete!")
print(f"📁 Saved at: {output_video}")
