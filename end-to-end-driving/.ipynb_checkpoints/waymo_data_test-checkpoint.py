import os, glob, inspect
import tensorflow as tf
from google.protobuf.message import Message, DecodeError

# the two modules you asked to use:
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2

# --- helpers ---
def list_proto_classes(mod):
    out = []
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and issubclass(obj, Message) and obj is not Message:
            out.append((name, obj))
    return out

def try_parse(rec_bytes, candidates):
    """Try parsing bytes with each candidate message class; return (name, msg) on first success."""
    for name, cls in candidates:
        msg = cls()
        try:
            msg.ParseFromString(rec_bytes)
            return name, msg
        except DecodeError:
            continue
    raise DecodeError("Record did not match any candidate message types.")

def open_tfrecord(path):
    """Open a TFRecord, trying uncompressed then GZIP."""
    try:
        ds = tf.data.TFRecordDataset(path, compression_type='')
        # touch one element to force I/O now; fall back to gzip if needed
        _ = next(iter(ds.take(1)))
        return ds, 'UNCOMPRESSED'
    except Exception:
        ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
        _ = next(iter(ds.take(1)))
        return ds, 'GZIP'

def get_field(frame, name):
    """Return (descriptor, value) or (None, None) if missing."""
    fd = frame.DESCRIPTOR.fields_by_name.get(name)
    if fd is None:
        return None, None
    return fd, getattr(frame, name)


# --- find a shard (edit BASE to your real folder) ---
BASE = "data/waymo-e2e/waymo_open_dataset_end_to_end_camera_v_1_0_0/"  # <- put your e2e root here
TRAIN_FILES = os.path.join(BASE, 'training.tfrecord*')
VALIDATION_FILES = os.path.join(BASE, 'validation.tfrecord*')
TEST_FILES = os.path.join(BASE, 'test.tfrecord*')

candidates = sorted(glob.glob(os.path.join(BASE, "**", "*.tfrecord*"), recursive=True))
if not candidates:
    raise FileNotFoundError(f"No TFRecords under {BASE}")

tfrecord_path = candidates[0]
ds, comp = open_tfrecord(tfrecord_path)
print(f"Opened: {tfrecord_path}\nCompression: {comp}")

# --- inspect available message classes and parse the first record ---
classes = list_proto_classes(wod_e2ed_pb2)
print("Candidate message types from end_to_end_driving_data_pb2:", [n for n,_ in classes])

first = next(iter(ds.take(1))).numpy()  # bytes
kind, msg = try_parse(first, classes)
print(f"Detected E2E message type: {kind}")

# --- example: print a few common fields if present (safe checks) ---
for field in ["run_id", "segment_id", "timestamp_us", "camera_images", "ego_state", "command"]:
    if msg.DESCRIPTOR.fields_by_name.get(field):
        val = getattr(msg, field)
        # repeated fields -> just show length
        if hasattr(val, "__len__") and not isinstance(val, (bytes, str)):
            print(field, "len:", len(val))
        else:
            print(field, ":", val)
            
# --- after you've parsed `frame` as E2EDFrame() ---
# handle intent / ego_intent robustly
for field_name in ("intent", "ego_intent"):
    fd, val = get_field(frame, field_name)
    if fd is None:
        continue

    # If it's an enum/scalar, print value (and enum name if available)
    if fd.type != fd.TYPE_MESSAGE:
        if fd.type == fd.TYPE_ENUM and fd.enum_type:
            enum_name = fd.enum_type.values_by_number.get(int(val))
            enum_name = enum_name.name if enum_name else str(val)
            print(f"{field_name}: {val} ({enum_name})")
        else:
            print(f"{field_name}: {val}")
    else:
        # It's a message → print its scalar fields
        msg = val
        scalars = [
            f.name for f in msg.DESCRIPTOR.fields
            if f.label != f.LABEL_REPEATED and f.type != f.TYPE_MESSAGE
        ]
        print(f"{field_name} scalars:", {n: getattr(msg, n) for n in scalars})
    break  # stop after first found field