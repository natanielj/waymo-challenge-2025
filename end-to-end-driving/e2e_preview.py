# e2e_preview_no_tf.py
import os, glob, inspect, hashlib
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from google.protobuf.message import Message
from google.protobuf.descriptor import FieldDescriptor
from tfrecord.reader import tfrecord_loader  # pure-Python TFRecord reader

# === CONFIG ===
BASE = "data/waymo-e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0"  # <- set your path
MAX_RECORDS = 500   # scan up to this many records
MAX_FRAMES  = 300   # cap buffered preview frames
MOSAIC_TILE = (2,3)

# === Waymo protos ===
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

def pick_shard(base):
    files = sorted(glob.glob(os.path.join(base, "**", "*.tfrecord*"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No TFRecords under {base}")
    order = ["training", "validation", "test"]
    for key in order:
        for p in files:
            if key in os.path.basename(p).lower():
                return p
    return files[0]

JPEG_SIG = b"\xFF\xD8"; PNG_SIG = b"\x89PNG\r\n\x1a\n"
def looks_like_img(b): return isinstance(b,(bytes,bytearray)) and (b.startswith(JPEG_SIG) or b.startswith(PNG_SIG))

def decode_img_bytes(b):
    arr = iio.imread(b)
    if arr.ndim == 2: arr = np.stack([arr]*3, -1)
    if arr.shape[-1] == 4: arr = arr[..., :3]
    return arr

def walk_for_images(msg, hints=None):
    """Return list[(cam_hint, np_img)] by recursively scanning for JPEG/PNG bytes."""
    if hints is None: hints = {}
    out = []
    if not hasattr(msg, "DESCRIPTOR"): return out
    cam_hint = hints.get("camera")
    for fd in msg.DESCRIPTOR.fields:
        name = fd.name
        val = getattr(msg, name)
        if fd.label == FieldDescriptor.LABEL_REPEATED:
            if fd.type == FieldDescriptor.TYPE_MESSAGE:
                for sub in val:
                    out.extend(walk_for_images(sub, hints=dict(hints)))
            elif fd.type == FieldDescriptor.TYPE_BYTES:
                for b in val:
                    if looks_like_img(b):
                        out.append((cam_hint or "CAM", decode_img_bytes(bytes(b))))
            else:
                if "camera" in name.lower() or "cam" in name.lower():
                    if len(val): hints["camera"] = str(val[-1])
        else:
            if fd.type == FieldDescriptor.TYPE_MESSAGE:
                out.extend(walk_for_images(val, hints=dict(hints)))
            elif fd.type == FieldDescriptor.TYPE_BYTES:
                if val and looks_like_img(val):
                    out.append((cam_hint or "CAM", decode_img_bytes(bytes(val))))
            else:
                if "camera" in name.lower() or "cam" in name.lower():
                    hints["camera"] = str(val)
    return out

def walk_for_first_int(msg, name_pred):
    if not hasattr(msg, "DESCRIPTOR"): return None
    for fd in msg.DESCRIPTOR.fields:
        name = fd.name
        val = getattr(msg, name)
        if fd.type in (FieldDescriptor.TYPE_INT32, FieldDescriptor.TYPE_INT64,
                       FieldDescriptor.TYPE_UINT32, FieldDescriptor.TYPE_UINT64,
                       FieldDescriptor.TYPE_FIXED32, FieldDescriptor.TYPE_FIXED64,
                       FieldDescriptor.TYPE_SFIXED32, FieldDescriptor.TYPE_SFIXED64,
                       FieldDescriptor.TYPE_SINT32, FieldDescriptor.TYPE_SINT64):
            if name_pred(name):
                try: return int(val)
                except: pass
        if fd.type == FieldDescriptor.TYPE_MESSAGE:
            if fd.label == FieldDescriptor.LABEL_REPEATED:
                for sub in val:
                    r = walk_for_first_int(sub, name_pred)
                    if r is not None: return r
            else:
                r = walk_for_first_int(val, name_pred)
                if r is not None: return r
    return None

def walk_for_first_str(msg, name_pred):
    if not hasattr(msg, "DESCRIPTOR"): return None
    for fd in msg.DESCRIPTOR.fields:
        name = fd.name
        val = getattr(msg, name)
        if fd.type == FieldDescriptor.TYPE_STRING and name_pred(name):
            s = str(val)
            if s: return s
        elif fd.type == FieldDescriptor.TYPE_BYTES and name_pred(name):
            try:
                s = bytes(val).decode("utf-8", errors="ignore")
                if s: return s
            except: pass
        if fd.type == FieldDescriptor.TYPE_MESSAGE:
            if fd.label == FieldDescriptor.LABEL_REPEATED:
                for sub in val:
                    r = walk_for_first_str(sub, name_pred)
                    if r is not None: return r
            else:
                r = walk_for_first_str(val, name_pred)
                if r is not None: return r
    return None

def best_segment_id(msg):
    s = walk_for_first_str(msg, lambda n: "segment_id" in n.lower() or n.lower()=="segment")
    if s: return s
    run = walk_for_first_str(msg, lambda n: "run_id" in n.lower() or n.lower()=="run") or "run_unknown"
    seg_idx = walk_for_first_int(msg, lambda n: "segment_index" in n.lower() or "segment_num" in n.lower())
    if seg_idx is not None: return f"{run}_seg{int(seg_idx):04d}"
    raw = (run or "") + (walk_for_first_str(msg, lambda n: "sequence" in n.lower()) or "")
    return f"{run}_seg_{hashlib.sha1(raw.encode()).hexdigest()[:8]}"

def best_timestamp_us(msg):
    ts = walk_for_first_int(msg, lambda n: "timestamp_us" in n.lower() or n.lower()=="time_us")
    if ts is not None: return int(ts)
    t = walk_for_first_int(msg, lambda n: "timestamp" in n.lower())
    if t is None: return 0
    t = int(t)
    if t > 1_000_000_000_000_000:  # ns
        return t // 1000
    if t > 1_000_000_000_000:      # ms
        return t * 1000
    return t

PREFERRED_CAM_ORDER = [
    "CAMERA_FRONT", "CAMERA_FRONT_LEFT", "CAMERA_FRONT_RIGHT",
    "CAMERA_SIDE_LEFT", "CAMERA_SIDE_RIGHT", "CAMERA_REAR",
]
def cam_sort_key(k):
    if k in PREFERRED_CAM_ORDER: return (0, PREFERRED_CAM_ORDER.index(k))
    if isinstance(k,str) and k.startswith("CAM_"):
        try: return (1, int(k.split("_",1)[1]))
        except: pass
    return (2, k)

def make_mosaic(frames_by_cam, tile=(2,3), pad=4):
    if not frames_by_cam: return None
    cams = sorted(frames_by_cam.keys(), key=cam_sort_key)
    Hmin = min(fr.shape[0] for fr in frames_by_cam.values())
    Wmin = min(fr.shape[1] for fr in frames_by_cam.values())
    def cc(im):
        H,W = im.shape[:2]
        y0 = max((H - Hmin)//2, 0); x0 = max((W - Wmin)//2, 0)
        return im[y0:y0+Hmin, x0:x0+Wmin]
    rows, cols = tile
    tiles=[]; it=iter(cams)
    for _ in range(rows):
        row=[]
        for _ in range(cols):
            k=next(it,None)
            row.append(np.zeros((Hmin,Wmin,3),dtype=np.uint8) if k is None else cc(frames_by_cam[k]))
        tiles.append(np.concatenate(row,1))
    out=np.concatenate(tiles,0)
    if pad: out=np.pad(out,((pad,pad),(pad,pad),(0,0)))
    return out

# ---- Open shard and proto ----
shard = pick_shard(BASE)
print("Opened shard:", shard)

E2EDFrame = None
for name, obj in inspect.getmembers(wod_e2ed_pb2):
    if inspect.isclass(obj) and issubclass(obj, Message) and name == "E2EDFrame":
        E2EDFrame = obj; break
assert E2EDFrame is not None, "E2EDFrame not found in your proto build"

# ---- Read records (UNCOMPRESSED expected; Waymo E2E is usually plain TFRecord) ----
records = []
count = 0
for rec in tfrecord_loader(shard, None, None):
    if count >= MAX_RECORDS: break
    count += 1
    m = E2EDFrame(); m.ParseFromString(rec)
    imgs = walk_for_images(m)
    if not imgs: continue
    by_cam = {}
    for cam, img in imgs:
        if cam not in by_cam: by_cam[cam] = img
    seg = best_segment_id(m)
    ts  = best_timestamp_us(m)
    records.append((seg, ts, by_cam))

if not records:
    raise RuntimeError("No JPEG/PNG image bytes found in this shard. Try a training/validation shard or verify you have the camera E2E bundle.")

# ---- Sort & build preview frames ----
records.sort(key=lambda r: (r[0], r[1]))
frames=[]; timeline=[]
for seg, ts, by_cam in records:
    frame = make_mosaic(by_cam, tile=MOSAIC_TILE) if len(by_cam)>1 else next(iter(by_cam.values()))
    if frame is not None:
        frames.append(frame); timeline.append((seg, ts))
    if len(frames) >= MAX_FRAMES: break

print(f"Buffered {len(frames)} frames (from {len(records)} records scanned).")

# ---- Preview (←/→ or p/n) ----
idx=0
fig, ax = plt.subplots(figsize=(10,6))
im = ax.imshow(frames[idx]); ax.axis('off')
seg, ts = timeline[idx]
ax.set_title(f"{seg}  ts={ts}  {idx+1}/{len(frames)}")
plt.tight_layout()

def on_key(ev):
    global idx
    if ev.key in ('n','right'): idx = min(idx+1, len(frames)-1)
    elif ev.key in ('p','left'): idx = max(idx-1, 0)
    else: return
    im.set_data(frames[idx])
    s,t = timeline[idx]
    ax.set_title(f"{s}  ts={t}  {idx+1}/{len(frames)}")
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
