# =========================================================
# ANIMAL-SPECIFIC TRACKING + TRAJECTORY + RISK + ACTION
# =========================================================

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================
# 1. LOAD MODEL
# =========================
model = YOLO(
    r"C:\Users\Dell\animal_tracking_dataset\runs\detect\train1\weights\best.pt"
)
print("Loaded classes:", model.names)

# =========================
# 2. VIDEO IO
# =========================
VIDEO_PATH= r"C:\Users\Dell\animal_tracking_dataset\raw_videos\cows\VID20251012183155.mp4"# change name if needed
OUTPUT_PATH  = r"C:\Users\Dell\animal_tracking_dataset\output_videos\final_o10.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# =========================
# 3. STORAGE
# =========================
trajectory_history = defaultdict(lambda: deque(maxlen=20))

# =========================
# 4. VEHICLE RISK ZONE
# =========================
risk_zone = np.array([
    (width//2 - 150, height),
    (width//2 + 150, height),
    (width//2 + 350, height//2),
    (width//2 - 350, height//2)
])

# =========================
# 5. TRAJECTORY PREDICTION
# =========================
def predict_future(traj, animal, steps=15):
    if len(traj) < 5:
        return []

    traj = np.array(traj)
    velocity = traj[1:] - traj[:-1]
    mean_velocity = velocity.mean(axis=0)

    if animal == "dog":
        scale, noise = 6.0, 20
    elif animal == "sheep":
        scale, noise = 4.0, 10
    else:  # cow
        scale, noise = 2.5, 4

    future = []
    last = traj[-1]

    for _ in range(steps):
        direction = mean_velocity * scale
        jitter = np.random.normal(0, noise, 2)
        next_pos = (last + direction + jitter).astype(int)
        future.append(tuple(next_pos))
        last = next_pos

    return future

# =========================
# 6. RISK CHECK
# =========================
def enters_risk_zone(predicted_points, zone):
    for pt in predicted_points:
        # Convert float → int (MANDATORY for OpenCV)
        x = int(pt[0])
        y = int(pt[1])

        if cv2.pointPolygonTest(zone, (x, y), False) >= 0:
            return True
    return False


# =========================
# 7. ACTION DECISION
# =========================
def decide_action(animal, danger):
    if not danger:
        return "Continue"

    if animal == "dog":
        return "Immediate Brake"
    elif animal == "cow":
        return "Gradual Slow Down"
    else:
        return "Alert + Slow"

# =========================
# 8. EXPLANATION (LLM-LIKE)
# =========================
def explain(animal, action, danger):
    if not danger:
        return "No risk detected. Vehicle continues safely."

    explanations = {
        "dog": "Dog movement is highly unpredictable, requiring immediate braking.",
        "cow": "Cow movement is slow but blocks the road, so gradual deceleration is safer.",
        "sheep": "Sheep may move in groups; early alert and reduced speed reduce risk."
    }
    return explanations.get(animal, "Caution advised due to animal movement.")

# =========================
# 9. MAIN LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        iou=0.5,
        tracker="bytetrack.yaml"
    )

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            cls_id = int(box.cls.item())
            animal = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            trajectory_history[track_id].append((cx, cy, cls_id))

            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
            cv2.putText(frame, f"{animal} ID:{track_id}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,255,0), 2)

    # =========================
    # DRAW TRAJECTORIES + ACTIONS
    # =========================
    for track_id, traj in trajectory_history.items():
        if len(traj) < 5:
            continue

        animal = model.names[traj[-1][2]]
        past = [(p[0], p[1]) for p in traj]

        for i in range(1, len(past)):
            cv2.line(frame, past[i-1], past[i], (0,255,0), 2)

        future = predict_future(past, animal)
        danger = enters_risk_zone(future, risk_zone)
        action = decide_action(animal, danger)
        explanation = explain(animal, action, danger)

        color = (0,0,255) if danger else (0,255,0)
        for i in range(1, len(future)):
            cv2.line(frame, future[i-1], future[i], color, 3)
            cv2.circle(frame, future[i], 3, color, -1)

        cv2.putText(frame, f"ACTION: {action}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255), 2)

    cv2.polylines(frame, [risk_zone], True, (255,0,0), 2)

    out.write(frame)
    cv2.imshow("Animal-Aware Trajectory Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# 10. CLEANUP
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ FINAL PIPELINE EXECUTED SUCCESSFULLY")
