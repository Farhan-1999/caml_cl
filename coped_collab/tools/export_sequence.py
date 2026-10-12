"""Export one CoPeD sequence (rosbags + released annotations) into per-frame files for training.

Output layout, one folder per robot:
    <out>/<SEQ>/<robot>/
        rgb/<stamp>.jpg          forward RealSense color, original JPEG bytes from the bag
        ir/<stamp>.jpg           left IR camera (infra1), original bytes
        ir_right/<stamp>.jpg     right IR camera (infra2), original bytes
        lidar/<stamp>.npz        ground robots: xyz (N,3) float32, intensity (N,) float32, ring (N,) uint8;
                                 points without a return (range 0, all-zero or non-finite) are dropped
        depth_zoe/<stamp>.png    ZoeDepth frame from the released depth video, 8-bit (brighter = closer)
        label/<stamp>.png        class-index mask: category id per pixel, 255 = unlabeled
        <modality>.csv           stamp_ns,path for every exported frame of that modality
        odom.csv                 stamp_ns,x,y,z,qx,qy,qz,qw from the robot's odometry
        camera_info.json         intrinsics of the exported cameras, when the bag has them
        tf_static.json           static transforms, when the bag has them
        export_summary.json      counts, time ranges and consistency checks

<stamp> is "<sec>_<nanosec>" from the message header, the naming the CoPeD labels already use,
so label, depth_zoe and rgb files of the same frame share a stamp. Other modalities have their
own stamps and are paired by nearest time when building samples.

Example:
    python export_sequence.py --root "E:/CoPed dataset" --sequence NYUARPL \
        --robots ptah race1 race5 wilbur --out "E:/CoPed dataset/exported"
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

AERIAL_ROBOTS = {"race1", "race5"}
UNLABELED = 255
MODALITIES = ("rgb", "ir", "ir_right", "lidar", "depth_zoe", "label")
BAG_MODALITIES = ("rgb", "ir", "ir_right", "lidar")

# Sensor topics per robot type, relative to "/<robot>/". Drones carry no LiDAR.
SENSOR_TOPICS = {
    "ground": {
        "rgb": "forward/color/image_rect_color/compressed",
        "ir": "forward/infra1/image_rect_raw/compressed",
        "ir_right": "forward/infra2/image_rect_raw/compressed",
        "lidar": "lidar_points",
    },
    "aerial": {
        "rgb": "cam1/color/image_raw/compressed",
        "ir": "cam1/infra1/image_rect_raw/compressed",
        "ir_right": "cam1/infra2/image_rect_raw/compressed",
    },
}
# Odometry topics, relative to "/<robot>/"; the first one present in the bag is used.
ODOM_TOPICS = {
    "ground": ("odom",),
    "aerial": ("odom_transform_nodelet/odomBinworld_from_transform", "mavros/local_position/odom"),
}
# sensor_msgs/PointField datatype codes
POINT_FIELD_DTYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def stamp_name(sec: int, nanosec: int) -> str:
    return f"{sec}_{nanosec:09d}"


def stamp_ns(name: str) -> int:
    sec, nsec = name.split("_")
    return int(sec) * 10**9 + int(nsec)


def field(msg, *names):
    """ROS1 messages use upper-case CameraInfo fields (K, D, ...); accept either spelling."""
    for name in names:
        if hasattr(msg, name):
            return getattr(msg, name)
    raise AttributeError(names)


def find_bag(root: Path, sequence: str, robot: str) -> Path:
    bags = [p for p in sorted((root / "sequence").glob(f"*/{sequence}/*.bag")) if robot in p.name]
    if len(bags) != 1:
        raise SystemExit(f"expected one bag for {sequence}/{robot}, found {[b.name for b in bags]}")
    return bags[0]


def write_index(rdir: Path, modality: str, rows: list[tuple[int, str]]) -> None:
    rows.sort()
    with open(rdir / f"{modality}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stamp_ns", "path"])
        writer.writerows(rows)


def time_range(rows: list[tuple[int, str]]) -> dict:
    if not rows:
        return {"count": 0}
    stamps = [r[0] for r in rows]
    return {"count": len(rows), "first_ns": min(stamps), "last_ns": max(stamps),
            "duration_s": round((max(stamps) - min(stamps)) / 1e9, 3)}


def pointcloud_arrays(msg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xyz, intensity and ring of the points that have a return."""
    if msg.is_bigendian:
        raise SystemExit("big-endian PointCloud2 is not supported")
    fields = [f for f in msg.fields if f.count == 1 and f.datatype in POINT_FIELD_DTYPES]
    dtype = np.dtype({"names": [f.name for f in fields],
                      "formats": [POINT_FIELD_DTYPES[f.datatype] for f in fields],
                      "offsets": [f.offset for f in fields],
                      "itemsize": msg.point_step})
    buf = msg.data.tobytes()
    if msg.row_step == msg.width * msg.point_step:
        pts = np.frombuffer(buf, dtype=dtype, count=msg.width * msg.height)
    else:  # rows padded to row_step
        pts = np.concatenate([np.frombuffer(buf, dtype=dtype, count=msg.width, offset=r * msg.row_step)
                              for r in range(msg.height)])
    xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float32)
    keep = np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)
    names = dtype.names
    if "range" in names:  # Ouster: range 0 = no return; such points sit at the beam origin, not at 0,0,0
        keep &= pts["range"] > 0
    intensity = pts["intensity"].astype(np.float32) if "intensity" in names else np.zeros(len(pts), np.float32)
    ring = pts["ring"].astype(np.uint8) if "ring" in names else np.zeros(len(pts), np.uint8)
    return xyz[keep], intensity[keep], ring[keep]


def export_bag(bag: Path, robot: str, rdir: Path, modalities: list[str], max_frames: int) -> dict:
    kind = "aerial" if robot in AERIAL_ROBOTS else "ground"
    wanted = {f"/{robot}/{SENSOR_TOPICS[kind][m]}": m
              for m in modalities if m in BAG_MODALITIES and m in SENSOR_TOPICS[kind]}
    typestore = get_typestore(Stores.ROS1_NOETIC)
    index = {m: [] for m in wanted.values()}
    seen = {m: set() for m in index}
    duplicates = {m: 0 for m in index}
    camera_info, tf_static, odom_rows = {}, {}, []
    lidar_frame, lidar_points = None, []

    with Reader(bag) as reader:
        sensor_conns = [c for c in reader.connections if c.topic in wanted]
        missing = set(wanted) - {c.topic for c in sensor_conns}
        if missing:
            raise SystemExit(f"{bag.name}: topics not found: {sorted(missing)}")
        # camera_info lives next to each image topic: /<robot>/forward/color/camera_info
        info_topics = {"/".join(t.split("/")[:-2]) + "/camera_info": m
                       for t, m in wanted.items() if t.endswith("/compressed")}
        info_conns = [c for c in reader.connections if c.topic in info_topics]
        tf_conns = [c for c in reader.connections if c.topic == "/tf_static"]
        odom_conns = []
        for suffix in ODOM_TOPICS[kind]:
            odom_conns = [c for c in reader.connections if c.topic == f"/{robot}/{suffix}"]
            if odom_conns:
                break
        # tf2_msgs is not in the default type store; register it from the bag's own definition
        for c in tf_conns:
            msgdef = c.msgdef.data if hasattr(c.msgdef, "data") else c.msgdef
            typestore.register(get_types_from_msg(msgdef, c.msgtype))

        for m in index:
            (rdir / m).mkdir(parents=True, exist_ok=True)
        for conn, _, raw in reader.messages(connections=sensor_conns + info_conns + tf_conns + odom_conns):
            if conn in sensor_conns:
                m = wanted[conn.topic]
                if max_frames and len(index[m]) >= max_frames:
                    if all(len(index[k]) >= max_frames for k in index):
                        break
                    continue
                msg = typestore.deserialize_ros1(raw, conn.msgtype)
                name = stamp_name(msg.header.stamp.sec, msg.header.stamp.nanosec)
                if name in seen[m]:
                    duplicates[m] += 1
                    continue
                seen[m].add(name)
                if m == "lidar":
                    xyz, intensity, ring = pointcloud_arrays(msg)
                    rel = f"lidar/{name}.npz"
                    np.savez(rdir / rel, xyz=xyz, intensity=intensity, ring=ring)
                    lidar_frame = lidar_frame or msg.header.frame_id
                    lidar_points.append(len(xyz))
                else:
                    ext = ".png" if "png" in msg.format.lower() else ".jpg"
                    rel = f"{m}/{name}{ext}"
                    (rdir / rel).write_bytes(msg.data.tobytes())
                index[m].append((stamp_ns(name), rel))
                continue
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            if conn in info_conns:
                m = info_topics[conn.topic]
                if m not in camera_info:
                    camera_info[m] = {
                        "topic": conn.topic,
                        "frame_id": msg.header.frame_id,
                        "width": msg.width,
                        "height": msg.height,
                        "distortion_model": msg.distortion_model,
                        "D": [float(x) for x in field(msg, "D", "d")],
                        "K": [float(x) for x in field(msg, "K", "k")],
                        "R": [float(x) for x in field(msg, "R", "r")],
                        "P": [float(x) for x in field(msg, "P", "p")],
                    }
            elif conn in tf_conns:
                for t in msg.transforms:
                    tr, rot = t.transform.translation, t.transform.rotation
                    tf_static[(t.header.frame_id, t.child_frame_id)] = {
                        "parent": t.header.frame_id, "child": t.child_frame_id,
                        "translation": [tr.x, tr.y, tr.z], "rotation_xyzw": [rot.x, rot.y, rot.z, rot.w],
                    }
            else:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                odom_rows.append((msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
                                  p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    for m, rows in index.items():
        write_index(rdir, m, rows)
    if camera_info:
        (rdir / "camera_info.json").write_text(json.dumps(camera_info, indent=2))
    if tf_static:
        (rdir / "tf_static.json").write_text(json.dumps(list(tf_static.values()), indent=2))
    if odom_rows:
        odom_rows.sort()
        with open(rdir / "odom.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw"])
            writer.writerows(odom_rows)
    summary = {
        "bag": bag.name,
        "modalities": {m: {**time_range(index[m]), "duplicate_stamps_skipped": duplicates[m]} for m in index},
        "camera_info": sorted(camera_info),
        "tf_static_transforms": len(tf_static),
        "odom": {"topic": odom_conns[0].topic, **time_range([(r[0], "") for r in odom_rows])} if odom_rows else None,
        "_index": index,
    }
    if lidar_points:
        summary["lidar"] = {"frame_id": lidar_frame, "points_per_scan_median": int(np.median(lidar_points)),
                            "points_per_scan_min": int(min(lidar_points))}
    return summary


def load_annotations(root: Path, sequence: str, robot: str) -> list[tuple[str, list]]:
    """(stem, segments) for every labelled frame, in time order; frame k of the videos is entry k."""
    data = json.loads((root / "groundtruth_semantics" / sequence / robot / "pred.json").read_text())
    return sorted((a["file_name"].rsplit(".", 1)[0], a["segments_info"]) for a in data["annotations"])


def export_labels(root: Path, sequence: str, robot: str, rdir: Path, annotations, workers: int) -> dict:
    src = root / "groundtruth_semantics" / sequence / robot / "Annotations"
    (rdir / "label").mkdir(parents=True, exist_ok=True)

    def convert(item):
        stem, segments = item
        im = cv2.imread(str(src / f"{stem}.png"), cv2.IMREAD_COLOR)
        if im is None:
            return stem, None, 0, 0
        # COCO-panoptic encoding: segment id = R + 256*G + 256^2*B (OpenCV loads BGR)
        ids = im[..., 2].astype(np.int32) | (im[..., 1].astype(np.int32) << 8) | (im[..., 0].astype(np.int32) << 16)
        mask = np.full(ids.shape, UNLABELED, np.uint8)
        for s in segments:
            mask[ids == s["id"]] = s["category_id"]
        unmatched = int(np.count_nonzero((ids != 0) & (mask == UNLABELED)))
        cv2.imwrite(str(rdir / "label" / f"{stem}.png"), mask)
        return stem, f"label/{stem}.png", unmatched, ids.size

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(convert, annotations))
    rows = [(stamp_ns(stem), rel) for stem, rel, _, _ in results if rel]
    write_index(rdir, "label", rows)
    pixels = sum(r[3] for r in results)
    categories = sorted({s["category_id"] for _, segs in annotations for s in segs})
    return {
        **time_range(rows),
        "missing_png": [stem for stem, rel, _, _ in results if rel is None][:20],
        "missing_png_count": sum(1 for _, rel, _, _ in results if rel is None),
        "unmatched_pixel_fraction": round(sum(r[2] for r in results) / max(1, pixels), 6),
        "category_ids": categories,
    }


def export_depth_video(root: Path, sequence: str, robot: str, rdir: Path, stems: list[str],
                       labelled_frames: int, workers: int) -> dict:
    """Frame k of the depth video is labelled frame k; `stems` may be a prefix (smoke test)."""
    videos = sorted((root / "video" / "depth_video").glob(f"{sequence}_{robot}_depth_*.mp4"))
    if len(videos) != 1:
        raise SystemExit(f"expected one depth video for {sequence}/{robot}, found {[v.name for v in videos]}")
    cap = cv2.VideoCapture(str(videos[0]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    (rdir / "depth_zoe").mkdir(parents=True, exist_ok=True)
    rows, read = [], 0
    with ThreadPoolExecutor(workers) as pool:
        futures = []
        for stem in stems:
            ok, frame = cap.read()
            if not ok:
                break
            read += 1
            rel = f"depth_zoe/{stem}.png"
            futures.append(pool.submit(cv2.imwrite, str(rdir / rel), frame[..., 0]))  # grayscale video
            rows.append((stamp_ns(stem), rel))
        for f in futures:
            f.result()
    cap.release()
    write_index(rdir, "depth_zoe", rows)
    return {**time_range(rows), "video": videos[0].name, "zoedepth_model": videos[0].stem.rsplit("_", 1)[-1],
            "video_frames": total, "labelled_frames": labelled_frames, "frames_read": read,
            "frame_count_matches_labels": total == labelled_frames}


def export_robot(args, robot: str) -> None:
    rdir = args.out / args.sequence / robot
    if rdir.exists():
        if not args.overwrite:
            raise SystemExit(f"{rdir} exists; pass --overwrite to replace it")
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True)
    t0 = time.time()
    summary = {"sequence": args.sequence, "robot": robot,
               "robot_type": "aerial" if robot in AERIAL_ROBOTS else "ground",
               "max_frames": args.max_frames, "unlabeled_value": UNLABELED}

    if any(m in BAG_MODALITIES for m in args.modalities):
        bag_info = export_bag(find_bag(args.root, args.sequence, robot), robot, rdir, args.modalities, args.max_frames)
        index = bag_info.pop("_index")
        summary["bag"] = bag_info
        print(f"  [{robot}] bag done in {time.time() - t0:.0f}s: "
              + ", ".join(f"{m}={len(v)}" for m, v in index.items()))
    else:
        index = {}

    annotations = load_annotations(args.root, args.sequence, robot)
    labelled_frames = len(annotations)
    if args.max_frames:
        annotations = annotations[: args.max_frames]
    stems = [stem for stem, _ in annotations]
    if "label" in args.modalities:
        summary["label"] = export_labels(args.root, args.sequence, robot, rdir, annotations, args.workers)
        print(f"  [{robot}] labels done: {summary['label']['count']} frames")
    if "depth_zoe" in args.modalities:
        summary["depth_zoe"] = export_depth_video(args.root, args.sequence, robot, rdir, stems,
                                                  labelled_frames, args.workers)
        print(f"  [{robot}] depth_zoe done: {summary['depth_zoe']['count']} frames")

    # Labels were made on the RGB stream, so every label stamp should be an RGB header stamp.
    if "rgb" in index:
        rgb = {s for s, _ in index["rgb"]}
        label_stamps = {stamp_ns(s) for s in stems}
        summary["checks"] = {
            "label_stamps_in_rgb": len(label_stamps & rgb),
            "label_stamps_total": len(label_stamps),
            "rgb_without_label": len(rgb - label_stamps),
        }
    summary["elapsed_s"] = round(time.time() - t0, 1)
    (rdir / "export_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  [{robot}] finished in {summary['elapsed_s']}s -> {rdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="CoPeD dataset root (groundtruth_semantics, sequence, video)")
    parser.add_argument("--sequence", required=True, help="e.g. NYUARPL")
    parser.add_argument("--robots", nargs="+", required=True, help="e.g. ptah race1 race5 wilbur")
    parser.add_argument("--out", type=Path, required=True, help="output root; <out>/<SEQ>/<robot>/ is created")
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES,
                        help="modalities a robot does not have (LiDAR on drones) are skipped")
    parser.add_argument("--max-frames", type=int, default=0, help="export only the first N frames per modality (smoke test)")
    parser.add_argument("--workers", type=int, default=8, help="threads for image writing")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing export for these robots")
    args = parser.parse_args()
    for robot in args.robots:
        print(f"exporting {args.sequence}/{robot}")
        export_robot(args, robot)


if __name__ == "__main__":
    main()
