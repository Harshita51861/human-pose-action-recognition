import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Try to import ipywidgets for notebook player (optional)
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    HAS_IPYWIDGETS = True
except Exception:
    HAS_IPYWIDGETS = False

# ====================== CONFIG ======================

# ------------- Paths -------------
VIDEO_PATH = "/kaggle/input/multiple/multiple.mp4"  # CHANGE IF NEEDED
OUT_DIR = "/kaggle/working"                         # where .npy files are saved

# ------------- YOLO & Pose -------------
MODEL_NAME = "yolov8n-pose.pt"  # lightweight pose model
CONF_THRESH = 0.25              # detection confidence threshold
KP_CONF_MIN = 0.30              # min avg keypoint confidence per person
MAX_PEOPLE = 5                  # store up to this many people per frame

# ------------- Video processing -------------
MAX_FRAMES = None   # None -> process whole video, or set e.g. 400

# ------------- Clip creation -------------
CLIP_LEN = 60       # number of frames per clip
STRIDE = 30         # step between consecutive clip starts
ACTION_LABEL = 0    # integer class label for this video

# ===================================================


def load_model(model_name: str) -> YOLO:
    """Load YOLOv8 pose model."""
    print(f"[INFO] Loading YOLOv8 model: {model_name}")
    model = YOLO(model_name)
    return model


def process_video_and_extract_keypoints(
    model: YOLO,
    video_path: str,
    max_frames: int | None = None,
    conf_thresh: float = 0.25,
    kp_conf_min: float = 0.30,
    max_people: int = 5,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Process the video frame-by-frame:
      - Run YOLO pose
      - Collect keypoints (x, y, conf) for each detected person
      - Filter low-confidence persons
      - Return:
          multi_skeleton: list of length T, each element (P_t, 17, 3)
          annotated_frames: list of RGB frames with skeletons drawn
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    multi_skeleton: List[np.ndarray] = []
    annotated_frames: List[np.ndarray] = []
    frame_idx = 0

    print(f"[INFO] Processing video: {video_path}")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        # Run YOLO pose on this frame
        results = model(frame, conf=conf_thresh, verbose=False)[0]

        # Draw skeletons for visualization (BGR -> RGB)
        annotated = results.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        annotated_frames.append(annotated)

        # Extract keypoints: (num_people, 17, 3)
        if results.keypoints is None:
            # No people in this frame
            multi_skeleton.append(
                np.zeros((0, 17, 3), dtype=np.float32)
            )
        else:
            kps = results.keypoints.data.cpu().numpy().astype(np.float32)
            filtered_people = []
            for person in kps:  # person: (17, 3)
                avg_conf = person[:, 2].mean()
                if avg_conf >= kp_conf_min:
                    filtered_people.append(person)

            if len(filtered_people) == 0:
                multi_skeleton.append(
                    np.zeros((0, 17, 3), dtype=np.float32)
                )
            else:
                multi_skeleton.append(
                    np.stack(filtered_people, axis=0)
                )

        if frame_idx % 50 == 0:
            print(f"[INFO] Processed {frame_idx} frames...")

    cap.release()
    elapsed = time.time() - start_time
    print(f"[INFO] Finished processing {frame_idx} frames in {elapsed:.2f}s")

    return multi_skeleton, annotated_frames


def pack_multi_person_tensor(
    multi_skeleton: List[np.ndarray],
    max_people: int,
    num_joints: int = 17,
) -> np.ndarray:
    """
    Convert list of per-frame person arrays into a fixed-size 4D tensor:
        (T, MAX_PEOPLE, J, C)
    Frames with fewer people are padded with zeros.
    """
    T = len(multi_skeleton)
    C = 3  # (x, y, conf)

    tensor = np.zeros((T, max_people, num_joints, C), dtype=np.float32)

    for t, persons in enumerate(multi_skeleton):
        if persons.shape[0] == 0:
            continue  # remains all zeros

        n = min(persons.shape[0], max_people)
        tensor[t, :n, :, :] = persons[:n]

    return tensor


def create_clips(
    skeleton_tensor: np.ndarray,
    clip_len: int,
    stride: int,
    action_label: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create temporal clips from the skeleton tensor using a sliding window.

    Input:
        skeleton_tensor: (T, MAX_PEOPLE, J, C)
    Output:
        X_clips: (N_clips, clip_len, MAX_PEOPLE, J, C)
        y_clips: (N_clips,) all filled with 'action_label'
    """
    T, max_people, J, C = skeleton_tensor.shape
    clips: List[np.ndarray] = []
    labels: List[int] = []

    if T <= clip_len:
        pad_len = clip_len - T
        pad = np.zeros((pad_len, max_people, J, C), dtype=skeleton_tensor.dtype)
        clip = np.concatenate([skeleton_tensor, pad], axis=0)
        clips.append(clip)
        labels.append(action_label)
    else:
        start = 0
        while start + clip_len <= T:
            clip = skeleton_tensor[start:start + clip_len]
            clips.append(clip)
            labels.append(action_label)
            start += stride

        # Optional: handle leftover frames at end
        if start < T:
            last_segment = skeleton_tensor[start:]
            pad_len = clip_len - last_segment.shape[0]
            if pad_len > 0:
                pad = np.zeros(
                    (pad_len, max_people, J, C),
                    dtype=skeleton_tensor.dtype
                )
                last_clip = np.concatenate([last_segment, pad], axis=0)
                clips.append(last_clip)
                labels.append(action_label)

    X_clips = np.stack(clips, axis=0)
    y_clips = np.array(labels, dtype=np.int64)
    return X_clips, y_clips


def build_normalized_adjacency_17j() -> np.ndarray:
    """
    Build normalized adjacency matrix for 17 COCO joints
    with the YOLOv8 pose skeleton definition.
    """
    NUM_JOINTS = 17

    # Edges for YOLOv8/COCO 17-keypoint skeleton
    EDGES = [
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (11, 12),
        (5, 11), (6, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (0, 5), (0, 6),
        (0, 1), (1, 3),
        (0, 2), (2, 4),
    ]

    A = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for i, j in EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # self connections
    for i in range(NUM_JOINTS):
        A[i, i] = 1.0

    # D^{-1/2} A D^{-1/2}
    D = A.sum(axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=D > 0)
    D_inv_sqrt[D <= 0] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    A_norm = D_mat @ A @ D_mat

    return A_norm


def save_numpy(array: np.ndarray, path: str, name: str) -> None:
    """Helper to save and print basic info."""
    np.save(path, array)
    print(f"[SAVE] {name} saved at: {path}")
    print(f"       shape = {array.shape}, dtype = {array.dtype}")


# ---------------------------------------------------------------------
# OPTIONAL: simple frame viewer + notebook player (if you import this
# file into a Jupyter/Kaggle notebook).
# ---------------------------------------------------------------------
def show_frame(frame: np.ndarray, title: str = "") -> None:
    """Show a single RGB frame with matplotlib."""
    plt.figure(figsize=(6, 4))
    plt.imshow(frame)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def show_player(annotated_frames: List[np.ndarray]) -> None:
    """
    Simple interactive player for annotated frames.
    Works only in notebook environments with ipywidgets.
    """
    if not HAS_IPYWIDGETS:
        print("[WARN] ipywidgets not available. Player cannot be shown.")
        return

    num_frames = len(annotated_frames)
    out_frame = widgets.Output()

    btn_play   = widgets.Button(description="▶ Play", button_style="success")
    btn_pause  = widgets.Button(description="⏸ Pause", button_style="warning")
    btn_prev   = widgets.Button(description="⏮ Prev")
    btn_next   = widgets.Button(description="⏭ Next")
    btn_restart= widgets.Button(description="⏹ Restart", button_style="danger")

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=num_frames - 1,
        step=1,
        description='Frame:',
        continuous_update=False,
        layout=widgets.Layout(width='80%')
    )

    controls = widgets.HBox([btn_play, btn_pause, btn_prev, btn_next, btn_restart])
    ui = widgets.VBox([controls, slider, out_frame])

    state = {"current_idx": 0, "playing": False}

    def _show(idx: int):
        idx = max(0, min(num_frames - 1, idx))
        frame = annotated_frames[idx]
        with out_frame:
            clear_output(wait=True)
            plt.figure(figsize=(6, 4))
            plt.imshow(frame)
            plt.axis("off")
            plt.title(f"Frame {idx+1}/{num_frames}")
            plt.show()

    def on_play_clicked(_):
        state["playing"] = True
        while state["playing"] and state["current_idx"] < num_frames:
            slider.value = state["current_idx"]
            _show(state["current_idx"])
            state["current_idx"] += 1
            time.sleep(0.04)

    def on_pause_clicked(_):
        state["playing"] = False

    def on_prev_clicked(_):
        state["current_idx"] = max(0, state["current_idx"] - 1)
        slider.value = state["current_idx"]
        _show(state["current_idx"])

    def on_next_clicked(_):
        state["current_idx"] = min(num_frames - 1, state["current_idx"] + 1)
        slider.value = state["current_idx"]
        _show(state["current_idx"])

    def on_restart_clicked(_):
        state["playing"] = False
        state["current_idx"] = 0
        slider.value = state["current_idx"]
        _show(state["current_idx"])

    def on_slider_change(change):
        if change["name"] == "value":
            state["playing"] = False
            state["current_idx"] = change["new"]
            _show(state["current_idx"])

    btn_play.on_click(on_play_clicked)
    btn_pause.on_click(on_pause_clicked)
    btn_prev.on_click(on_prev_clicked)
    btn_next.on_click(on_next_clicked)
    btn_restart.on_click(on_restart_clicked)
    slider.observe(on_slider_change)

    display(ui)
    _show(0)


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load YOLO model
    model = load_model(MODEL_NAME)

    # 2) Process video -> keypoints + annotated frames
    multi_skeleton, annotated_frames = process_video_and_extract_keypoints(
        model=model,
        video_path=VIDEO_PATH,
        max_frames=MAX_FRAMES,
        conf_thresh=CONF_THRESH,
        kp_conf_min=KP_CONF_MIN,
        max_people=MAX_PEOPLE,
    )

    print(f"[INFO] Total frames processed: {len(multi_skeleton)}")

    # 3) Pack into fixed-size tensor (T, MAX_PEOPLE, 17, 3)
    skeleton_tensor = pack_multi_person_tensor(
        multi_skeleton=multi_skeleton,
        max_people=MAX_PEOPLE,
        num_joints=17,
    )

    # Optional debug
    frame0 = skeleton_tensor[0]
    print("[DEBUG] First frame skeleton tensor shape:", frame0.shape)

    # 4) Save skeleton tensor
    skel_path = os.path.join(OUT_DIR, "skeleton_data.npy")
    save_numpy(skeleton_tensor, skel_path, "Skeleton tensor")

    # 5) Create clips
    X_clips, y_clips = create_clips(
        skeleton_tensor=skeleton_tensor,
        clip_len=CLIP_LEN,
        stride=STRIDE,
        action_label=ACTION_LABEL,
    )

    X_path = os.path.join(OUT_DIR, "X_clips.npy")
    y_path = os.path.join(OUT_DIR, "y_clips.npy")
    save_numpy(X_clips, X_path, "X_clips")
    save_numpy(y_clips, y_path, "y_clips")

    # 6) Adjacency matrix for GCN/ST-GCN
    A_norm = build_normalized_adjacency_17j()
    adj_path = os.path.join(OUT_DIR, "adjacency_17j.npy")
    save_numpy(A_norm, adj_path, "Adjacency matrix (normalized)")

    print("[DONE] Full skeleton pipeline finished.")


if __name__ == "__main__":
    main()
