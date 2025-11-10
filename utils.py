import os
import json
import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
from typing import Dict, Tuple, Optional

# ---------------- Optional: GPU memory growth ----------------
# Aman dipanggil meskipun tidak ada GPU
try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
except Exception:
    pass


# ---------------- Public API: load_label_map ----------------
def load_label_map(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Membaca label map JSON, memastikan key menjadi int.
    Return:
      idx2name: {int_index: "label_name"}
      name2idx: {"label_name": int_index}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label map tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Normalisasi key ke int (dukung file yang tersimpan sebagai string key)
    idx2name = {int(k): str(v) for k, v in raw.items()}
    name2idx = {v: k for k, v in idx2name.items()}
    return idx2name, name2idx


# ---------------- Public API: load_keras_model ----------------
def load_keras_model(
    model_dir: str,
    custom_objects: Optional[dict] = None,
):
    """
    Load model dari {model_dir}/best_fer3.keras atau {model_dir}/best_fer3.h5.
    Selalu compile=False agar bebas dari optimizer/lr-schedule custom (mis. WarmupCosine).
    """
    h5_path = os.path.join(model_dir, "best_fer3.h5")
    keras_path = os.path.join(model_dir, "best_fer3.keras")

    # Urutan prefer: .keras → .h5
    candidates = []
    if os.path.exists(keras_path):
        candidates.append(keras_path)
    if os.path.exists(h5_path):
        candidates.append(h5_path)

    if not candidates:
        raise FileNotFoundError(
            f"Model tidak ditemukan di {model_dir} (butuh best_fer3.keras atau best_fer3.h5)"
        )

    last_exc = None
    for p in candidates:
        try:
            print("[Model] Loading:", p)
            return tf.keras.models.load_model(p, compile=False, custom_objects=custom_objects)
        except Exception as e:
            last_exc = e
            print("[Model] Gagal load:", p, "->", repr(e))
            continue

    # Kalau dua-duanya gagal:
    raise RuntimeError(f"Gagal memuat model. Terakhir error: {last_exc}")


# ---------------- Public API: preprocess_face_for_model ----------------
def preprocess_face_for_model(
    face,
    target_size: Tuple[int, int] = (48, 48),
    use_clahe: bool = False,
) -> np.ndarray:
    """
    Ubah ROI wajah menjadi (1, H, W, 1) float32 [0..1] untuk inference.
    - face dapat berupa grayscale (H, W) atau BGR (H, W, 3).
    - use_clahe=False (default) agar sesuai data training;
      bisa set True kalau ingin boosting kontras (kadang membantu kelas 'sad').
    """
    if face is None or getattr(face, "size", 0) == 0:
        H, W = target_size[1], target_size[0]
        return np.zeros((1, H, W, 1), dtype=np.float32)

    # Pastikan grayscale
    if face.ndim == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # (opsional) CLAHE untuk kontras
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face = clahe.apply(face)

    # Resize ke ukuran model
    face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)

    # Normalisasi 0..1, pastikan contiguous
    face = np.ascontiguousarray(face, dtype=np.float32) / 255.0

    # Tambah channel & batch
    face = np.expand_dims(face, axis=-1)  # (H, W, 1)
    face = np.expand_dims(face, axis=0)   # (1, H, W, 1)
    return face


# ---------------- Public API: iou ----------------
def iou(boxA, boxB) -> float:
    """
    IoU antara dua box (x,y,w,h). Mengembalikan 0.0 bila box None.
    """
    if boxA is None or boxB is None:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(1, boxA[2] * boxA[3])
    boxBArea = max(1, boxB[2] * boxB[3])

    return float(interArea) / float(boxAArea + boxBArea - interArea + 1e-6)


# ---------------- Public API: FaceTrack ----------------
class FaceTrack:
    """
    Tracking + stabilisasi label emosi:
      - EMA untuk posisi box (update_box)
      - Voting mayoritas untuk label (pred_queue)
      - EMA probability untuk confidence yang lebih stabil (prob_ema)
    Kompatibel dengan main.py: add_pred(label, probs=None)
    """
    def __init__(self, box, ema_alpha: float = 0.45, pred_win: int = 25, pred_thr: int = 18):
        self.box = box
        self.ema_alpha = float(ema_alpha)
        self.miss = 0

        # Smoothing label
        self.pred_queue = deque(maxlen=int(pred_win))
        self.pred_thr = int(pred_thr)
        self.stable_label = None
        self.last_shown_label = None

        # EMA probability
        self.prob_ema = None  # np.ndarray same shape as model probs

        # opsional id (diisi dari main)
        self.id = None

    def update_box(self, new_box):
        if self.box is None:
            self.box = new_box
            self.miss = 0
            return

        bx, by, bw, bh = self.box
        nx, ny, nw, nh = new_box
        a = self.ema_alpha

        self.box = (
            int(a * nx + (1 - a) * bx),
            int(a * ny + (1 - a) * by),
            int(a * nw + (1 - a) * bw),
            int(a * nh + (1 - a) * bh),
        )
        self.miss = 0

    def add_pred(self, label: str, probs: Optional[np.ndarray] = None):
        """
        Tambah prediksi frame:
          label : 'happy' | 'sad' | 'neutral'
          probs : vector probabilitas (opsional); di-EMA untuk confidence stabil
        Return:
          (is_stable: bool, top: str, count: int)
        """
        self.pred_queue.append(label)

        # EMA probs
        if probs is not None:
            p = np.asarray(probs, dtype=np.float32)
            if self.prob_ema is None:
                self.prob_ema = p
            else:
                # EMA dengan koefisien sama a (cukup stabil)
                a = self.ema_alpha
                self.prob_ema = a * p + (1.0 - a) * self.prob_ema

        # Voting mayoritas
        cnt = Counter(self.pred_queue)
        top, count = cnt.most_common(1)[0]
        if count >= self.pred_thr:
            if top != self.stable_label:
                self.stable_label = top
            return True, top, count
        return False, top, count


# ---------------- Public API: read_training_log ----------------
def read_training_log(model_dir: str) -> Optional[dict]:
    """
    Baca baris terakhir training_log.csv (jika ada), kembalikan dict ringkas.
    """
    log_path = os.path.join(model_dir, "training_log.csv")
    if not os.path.exists(log_path):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(log_path)
        if df.empty:
            return None
        last = df.iloc[-1].to_dict()
        # pastikan kunci utama tersedia kalau ada
        return {
            "epoch": last.get("epoch"),
            "accuracy": last.get("accuracy"),
            "val_accuracy": last.get("val_accuracy"),
        }
    except Exception as e:
        print("[Log] gagal dibaca:", e)
        return None


# ---------------- Optional: deterministik (kalau perlu) ----------------
def set_seed(seed: int = 42):
    """
    Opsional: panggil ini di awal program jika ingin hasil lebih deterministik.
    Note: determinisme penuh di TF bisa mempengaruhi performa.
    """
    import random, numpy
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    numpy.random.seed(seed)
    tf.random.set_seed(seed)
