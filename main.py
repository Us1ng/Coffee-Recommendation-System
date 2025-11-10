import os, time, cv2, numpy as np
from collections import OrderedDict

from utils import (
    load_label_map, load_keras_model, preprocess_face_for_model,
    FaceTrack, iou, read_training_log
)
from recommend import get_image_recommendations, RecoMemory

# ================== PATH ==================
CASCADE_PATH   = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")
MODEL_DIR      = "model"
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# ============ DETECTOR/TRACKER ============
SCALE_FACTOR    = 1.06
MIN_NEIGHBORS   = 6
MIN_FACE_SIZE   = (70, 70)
DETECT_INTERVAL = 5
MAX_MISS        = 10

# ============== SMOOTHING =================
# (label voting di FaceTrack. Stabil 4s dikontrol di MenuState.)
PRED_WINDOW    = 21  # ← dilonggarkan
PRED_THRESHOLD = 14  # ← dilonggarkan (~67%)

# ============= STABILITY RULES ============
MIN_STABLE_SECS          = 4.0     # emosi stabil minimal 4 detik
MENU_COOLDOWN_SECS       = 10.0    # jeda minimal menu baru
MIN_CONFIDENCE_TO_CHANGE = 0.70    # confidence minimal 70%

# Hysteresis khusus neutral <-> sad agar tidak gampang saling ganti
SAD_NEU_EXTRA_SECS  = 0.8   # ← diperingan
SAD_NEU_EXTRA_CONF  = 0.02  # ← diperingan (+2%)

# ================== OVERLAY =================
OVERLAY_X, OVERLAY_Y = 10, 10
OVERLAY_ALPHA = 0.90
MAX_ITEMS = 3
HEADER_BG = (18,18,18)
EMO_COLOR = {
    "happy":   (60, 180, 75),
    "neutral": (200, 160, 60),
    "sad":     (70, 70, 200),
    "default": (18, 18, 18),
}
# Overlay diperkecil & rapi
TILE_W, TILE_H = 120, 100
TILE_GAP = 6
HEADER_H = 28
BOTTOM_TEXT_H = 20

# ================= WINDOW ==================
WINDOW_TITLE = "Coffee Mood Recommender — pro1"
FULLSCREEN_DEFAULT = False  # fleksibel (bisa resize)

# --------------- UI Helpers ---------------
def draw_transparent_img(bg, overlay, x, y, alpha=0.9):
    h, w = overlay.shape[:2]
    H, W = bg.shape[:2]
    if x >= W or y >= H:
        return bg
    w = min(w, W - x); h = min(h, H - y)
    roi = bg[y:y+h, x:x+w]
    ov  = overlay[:h, :w]
    blended = cv2.addWeighted(roi, 1-alpha, ov, alpha, 0)
    bg[y:y+h, x:x+w] = blended
    return bg

def load_and_resize(path, size=(TILE_W, TILE_H)):
    # Cek None eksplisit (hindari "array or default" yg bikin error)
    if not os.path.exists(path):
        img = np.full((size[1], size[0], 3), 60, np.uint8)
        cv2.putText(img, "No Image", (6, size[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        return img
    img = cv2.imread(path)
    if img is None:
        img = np.full((size[1], size[0], 3), 100, np.uint8)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def text_strip(img, rect, text, scale=0.5):
    """Taruh strip teks gelap (anti bentrok) di dalam rect (x,y,w,h)."""
    x,y,w,h = rect
    strip_h = 18
    y0 = y + h - strip_h
    cv2.rectangle(img, (x, y0), (x+w, y+h), (20,20,20), -1)
    if len(text) > 16: text = text[:15] + "…"
    cv2.putText(img, text, (x+4, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255,255,255), 1, cv2.LINE_AA)

def render_overlay(label, items):
    """Overlay rapi: header kecil + tile dengan strip teks masing-masing."""
    pad = 8
    n = min(len(items or []), MAX_ITEMS)
    panel_w = pad*2 + n*TILE_W + (n-1)*TILE_GAP
    panel_h = pad*2 + HEADER_H + TILE_H + BOTTOM_TEXT_H
    panel = np.full((panel_h, panel_w, 3), HEADER_BG, np.uint8)

    # header warna emosi
    color = EMO_COLOR.get(label, EMO_COLOR["default"])
    cv2.rectangle(panel, (pad, pad), (panel_w-pad, pad+HEADER_H), color, -1)
    cv2.putText(panel, f"Recommended ({label if label else '-'})",
                (pad+6, pad+HEADER_H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255,255,255), 1, cv2.LINE_AA)

    # tiles
    y_img = pad + HEADER_H
    for i in range(n):
        name, path = items[i]
        xi = pad + i*(TILE_W + TILE_GAP)
        panel[y_img:y_img+TILE_H, xi:xi+TILE_W] = load_and_resize(path)
        text_strip(panel, (xi, y_img, TILE_W, TILE_H), name, scale=0.5)
    return panel

def draw_face_box_with_label(frame, box, label):
    """Kotak wajah + hanya tulisan gesture (tanpa persen), dengan background label."""
    x,y,w,h = box
    # kotak
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,180,255), 2)
    # label background
    lbl = label if label else "-"
    (tw,th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    pad = 4
    y0 = max(0, y - th - 2*pad)
    cv2.rectangle(frame, (x, y0), (x+tw+2*pad, y0+th+2*pad), (0,0,0), -1)
    cv2.putText(frame, lbl, (x+pad, y0+th+pad), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,255), 2, cv2.LINE_AA)

# ---------- Fokus wajah terbesar ----------
def choose_active_id(tracks):
    if not tracks:
        return None
    best_id, best_area = None, 0
    for tid, tr in tracks.items():
        if getattr(tr, "box", None) is None:
            continue
        x,y,w,h = tr.box
        area = w * h
        if area > best_area:
            best_id, best_area = tid, area
    return best_id

# -------- Menu state + hysteresis ----------
class MenuState:
    def __init__(self, cooldown_s=MENU_COOLDOWN_SECS,
                 min_stable_s=MIN_STABLE_SECS,
                 min_conf=MIN_CONFIDENCE_TO_CHANGE):
        self.current_label = None        # label yang sedang ditampilkan di overlay
        self.current_items = []
        self.last_change = 0.0           # kapan overlay terakhir berubah
        self.last_stable_label = None    # label stabil terbaru dari tracker
        self.last_stable_since = 0.0     # sejak kapan label stabil ini bertahan
        self.cooldown_s = cooldown_s
        self.min_stable_s = min_stable_s
        self.min_conf = min_conf

    def _is_sad_neutral_transition(self, new_label):
        if self.current_label in ("sad","neutral") and new_label in ("sad","neutral"):
            return self.current_label != new_label
        return False

    def update_stable(self, stable_label, conf):
        now = time.time()

        # track durasi stabilnya label (berdasarkan stable label dari tracker)
        if stable_label != self.last_stable_label:
            self.last_stable_label = stable_label
            self.last_stable_since = now
        stable_duration = now - self.last_stable_since

        # base rules
        cooldown_ok = (now - self.last_change) >= self.cooldown_s
        stable_ok   = stable_duration >= self.min_stable_s
        conf_ok     = (conf is None) or (conf >= self.min_conf)

        # hysteresis untuk transisi sad <-> neutral
        if self._is_sad_neutral_transition(stable_label):
            stable_ok = stable_duration >= (self.min_stable_s + SAD_NEU_EXTRA_SECS)
            if conf is not None:
                conf_ok = conf >= (self.min_conf + SAD_NEU_EXTRA_CONF)

        should_change = (
            stable_label is not None and
            stable_label != self.current_label and
            cooldown_ok and stable_ok and conf_ok
        )
        return should_change

    def set_menu(self, label, items):
        self.current_label = label
        self.current_items = items or []
        self.last_change = time.time()

# ================== MAIN ==================
def main():
    idx2name, name2idx = load_label_map(LABEL_MAP_PATH)
    model = load_keras_model(MODEL_DIR)
    print("[Model] Loaded classes:", idx2name)
    info = read_training_log(MODEL_DIR)
    if info:
        print(f"[Training] epoch={info.get('epoch')} acc={info.get('accuracy')} val_acc={info.get('val_accuracy')}")

    emotion_from_idx = lambda i: idx2name.get(int(i), "unknown")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError("Gagal load haarcascade.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # window fleksibel
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 960, 540)

    tracks = OrderedDict()
    next_id = 1
    frame_idx = 0

    menu = MenuState()
    reco_mem = RecoMemory(maxlen=10, cooldown_s=MENU_COOLDOWN_SECS)

    active_id = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------- deteksi & sinkronisasi track (tiap N frame) --------
        if frame_idx % DETECT_INTERVAL == 0:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=MIN_FACE_SIZE,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            det_boxes = list(faces) if len(faces) else []

            used = set()
            for tid, tr in list(tracks.items()):
                cur = getattr(tr, "box", None)
                best_i, best_iou = -1, 0.0
                for i, b in enumerate(det_boxes):
                    if i in used: continue
                    ov = iou(cur, b)
                    if ov > best_iou:
                        best_iou, best_i = ov, i
                if best_i >= 0 and best_iou >= 0.2:
                    bbox = det_boxes[best_i]
                    if hasattr(tr, "update_box"):
                        tr.update_box(bbox)
                    else:
                        tr.box = bbox
                        tr.miss = 0
                    used.add(best_i)
                else:
                    tr.miss = getattr(tr, "miss", 0) + 1
                    if tr.miss > MAX_MISS:
                        tracks.pop(tid, None)

            for i, b in enumerate(det_boxes):
                if i in used: continue
                tr = FaceTrack(box=b, ema_alpha=0.45, pred_win=PRED_WINDOW, pred_thr=PRED_THRESHOLD)
                tr.id = next_id
                tracks[next_id] = tr
                next_id += 1

        # -------- fokus: pilih wajah TERBESAR --------
        active_id = choose_active_id(tracks)

        stable_for_menu, stable_conf = None, None

        # -------- prediksi emosi pada wajah aktif --------
        if active_id and active_id in tracks:
            tr = tracks[active_id]
            if tr.box is not None:
                x,y,w,h = tr.box
                # ROI wajah (grayscale)
                face_roi = gray[y:y+h, x:x+w]
                if face_roi.size > 0:
                    # CLAHE ON untuk bantu sad
                    inp = preprocess_face_for_model(face_roi, use_clahe=True)  # (1,48,48,1)
                    probs = model.predict(inp, verbose=0)[0]   # e.g., (3,)

                    # ==== PRO-SAD CALIBRATION (ringan) ====
                    prob = probs.astype(np.float32).copy()
                    if prob.shape[0] >= 2:
                        prob[1] *= 1.08  # naikan peluang 'sad' 8% relatif
                    s = float(prob.sum())
                    if s > 0: prob /= s
                    # ======================================

                    pred_idx = int(np.argmax(prob))
                    pred_emo = emotion_from_idx(pred_idx)

                    # update smoothing label + prob (pakai prob hasil kalibrasi)
                    is_stable, top, cnt = tr.add_pred(pred_emo, probs=prob)

                    # Tampilkan hanya kotak + gesture (tanpa persen)
                    gesture_to_show = tr.stable_label if tr.stable_label else pred_emo
                    draw_face_box_with_label(frame, tr.box, gesture_to_show)

                    # syarat stabil + kelas valid
                    if is_stable and top in ("happy","sad","neutral"):
                        p_ema = getattr(tr, "prob_ema", None)
                        stable_for_menu = top
                        stable_conf = float(np.max(p_ema)) if p_ema is not None else float(np.max(prob))

        # -------- update menu (cooldown 10s + stabil 4s + conf≥70% + hysteresis sad/neutral) --------
        if stable_for_menu:
            if menu.update_stable(stable_for_menu, stable_conf):
                if stable_conf is None or stable_conf >= MIN_CONFIDENCE_TO_CHANGE:
                    if not reco_mem.was_recent(stable_for_menu):
                        picks = get_image_recommendations(stable_for_menu, k=MAX_ITEMS)
                        menu.set_menu(stable_for_menu, picks)
                        reco_mem.push(stable_for_menu, picks)

        # -------- overlay kecil di pojok kiri atas --------
        panel = render_overlay(menu.current_label, menu.current_items)
        frame = draw_transparent_img(frame, panel, OVERLAY_X, OVERLAY_Y, alpha=OVERLAY_ALPHA)

        cv2.imshow(WINDOW_TITLE, frame)
        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
