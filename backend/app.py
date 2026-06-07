from __future__ import annotations

import base64
import csv
import io
import json
import os
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"
RUNS_DIR = PROJECT_DIR / "runs"
ULTRA_DIR = BASE_DIR / ".ultralytics"

for directory in (UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR, ULTRA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRA_DIR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(PROJECT_DIR))

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file, send_from_directory
from ultralytics import YOLO


DEFAULT_MODEL_CANDIDATES = (
    PROJECT_DIR / "models" / "YOLO11n-seg.pt",
    PROJECT_DIR / "models" / "corrosion" / "YOLO11n-seg.pt",
)
DEFAULT_MODEL = next((path for path in DEFAULT_MODEL_CANDIDATES if path.exists()), DEFAULT_MODEL_CANDIDATES[0])
CLASS_NAMES = {
    0: "缝隙腐蚀",
    1: "点蚀",
    2: "均匀腐蚀",
}
CLASS_EN = {
    0: "Crevice Corrosion",
    1: "Pitting Corrosion",
    2: "Uniform Corrosion",
}
CLASS_KEYS = {
    0: "crevice",
    1: "pitting",
    2: "uniform",
}
REPORT_MAINTENANCE_GUIDES = {
    "crevice": {
        "title": "缝隙腐蚀（法兰 / 搭接面 / 垫片区）",
        "pain": "缝隙内介质滞留、隐蔽性强、易反复渗漏。",
        "groups": [
            {
                "title": "可拆卸结构（法兰 / 螺栓 / 换热器管板）",
                "items": [
                    "拆解修复：解体→喷砂至 Sa2.5 级→更换耐蚀垫片（如 PTFE、改性橡胶）→螺栓涂防咬合剂（如二硫化钼）→均匀紧固。",
                    "缝隙填充：用无收缩环氧密封胶 / 聚硫橡胶填充缝隙，阻断介质渗入。",
                ],
            },
            {
                "title": "不可拆结构（焊缝 / 搭接缝）",
                "items": [
                    "表面处理：打磨缝隙至金属本色→高压水 + 化学清洗，去除氯离子与沉积物。",
                    "涂层密封：刷涂玻璃鳞片涂料 / 碳纳米聚合物材料（如索雷 SD8000），形成无缝保护层。",
                    "焊补优化：连续满焊替代间断焊，消除微缝隙；焊后酸洗钝化。",
                ],
            },
            {
                "title": "长效预防",
                "items": [
                    "设计：减少搭接 / 死角，采用全焊透结构。",
                    "材质：选用双相钢 2205/2507、高钼不锈钢（如 316L）。",
                    "运维：定期紧固螺栓、清洗沉积物，避免垢下腐蚀。",
                ],
            },
        ],
    },
    "pitting": {
        "title": "点蚀（不锈钢 / 碳钢局部深坑，易穿孔）",
        "pain": "蚀孔小而深、隐蔽性强、易突发泄漏。",
        "groups": [
            {
                "title": "轻度点蚀（深度≤0.5mm，无穿孔）",
                "items": [
                    "打磨钝化：机械打磨至蚀孔消失→酸洗钝化（不锈钢）→抛光至 Ra≤1.6μm。",
                    "腻子填补：环氧腻子填充小坑→打磨平整→涂防腐涂层。",
                ],
            },
            {
                "title": "中度点蚀（0.5mm＜深度＜壁厚 20%）",
                "items": [
                    "补焊修复：匹配材质焊条（如 316L 用 E316L）→小电流堆焊→打磨光滑→无损检测（PT/MT）。",
                    "冷喷 / 熔覆：冷喷锌 / 镍基合金，热影响区小，适合精密部件。",
                ],
            },
            {
                "title": "重度点蚀（穿孔 / 深度≥壁厚 20%）",
                "items": [
                    "局部贴板：蚀孔周边打磨→贴耐蚀钢板（如 316L）→连续满焊→防腐处理。",
                    "整体更换：关键 / 高压设备，优先更换部件，避免安全风险。",
                ],
            },
            {
                "title": "长效预防",
                "items": [
                    "材质：选用高钼不锈钢（316L）、双相钢、钛合金。",
                    "介质：控制氯离子≤50ppm，添加缓蚀剂（如铬酸盐、有机胺）。",
                    "表面：提高光洁度，避免划伤与沉积物堆积。",
                ],
            },
        ],
    },
    "uniform": {
        "title": "均匀腐蚀（大面积整体减薄，如碳钢大气 / 土壤腐蚀）",
        "pain": "壁厚均匀减薄、强度下降、寿命缩短。",
        "groups": [
            {
                "title": "轻度腐蚀（减薄≤5%，无坑蚀）",
                "items": [
                    "表面清理：手工 / 动力工具除锈至 St3 级→脱脂（丙酮 / 乙醇）→干燥。",
                    "大气 / 淡水：环氧底漆 + 聚氨酯面漆，总厚≥200μm。",
                    "土壤 / 埋地：环氧煤沥青 + 玻璃纤维布，加强级防腐。",
                    "高温 / 化工：玻璃鳞片涂料、耐蚀砖衬里。",
                ],
            },
            {
                "title": "中度腐蚀（5%＜减薄＜10%，局部锈坑）",
                "items": [
                    "壁厚补强：腐蚀区贴同材质钢板→满焊→防腐涂层；或碳纤维复合材料补强（不动火）。",
                    "热喷涂：电弧喷涂锌 / 铝，厚度≥100μm，牺牲阳极保护。",
                ],
            },
            {
                "title": "重度腐蚀（减薄≥10% / 强度不足）",
                "items": [
                    "整体更换：重要设备（如压力容器、管道），更换为耐蚀材质（如不锈钢、玻璃钢）。",
                    "衬里修复：内壁衬橡胶、PTFE、玻璃钢，隔离腐蚀介质。",
                ],
            },
            {
                "title": "长效预防",
                "items": [
                    "设计：增加腐蚀余量（如碳钢≥2mm），定期超声波测厚。",
                    "阴极保护：埋地 / 水下结构，牺牲阳极（镁 / 锌）或外加电流保护。",
                    "材质升级：换用耐候钢、不锈钢、塑料 / 玻璃钢等。",
                ],
            },
        ],
    },
}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")
model: YOLO | None = None
model_path = DEFAULT_MODEL
reports: dict[str, dict[str, Any]] = {}
MODEL_SUFFIXES = {".pt", ".onnx", ".engine"}
MODEL_FPS_CACHE: dict[str, str] = {}
BENCHMARK_IMAGE = np.zeros((640, 640, 3), dtype=np.uint8)
VIDEO_DETECT_STRIDE = 2


def load_model(path: Path | str | None = None) -> YOLO:
    global model, model_path
    target = Path(path) if path else model_path
    if not target.exists():
        raise FileNotFoundError(f"模型文件不存在: {target}")
    model_path = target
    model = YOLO(str(target))
    return model


def get_model() -> YOLO:
    global model
    if model is None:
        model = load_model(model_path)
    return model


def resolved_key(path: Path | str) -> str:
    return str(Path(path).resolve()).lower()


def candidate_results_csv(path: Path) -> Path | None:
    resolved = path.resolve()
    default_results = RUNS_DIR / "segment" / "train" / "results.csv"
    if resolved == DEFAULT_MODEL.resolve():
        return default_results

    parts = [part.lower() for part in resolved.parts]
    if "runs" in parts and resolved.parent.name.lower() == "weights":
        run_results = resolved.parent.parent / "results.csv"
        if run_results.exists():
            return run_results

    return None


def format_map50(value: Any) -> str:
    if value in (None, ""):
        return "--"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "--"


def checkpoint_map50(path: Path | str) -> str:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix.lower() != ".pt":
        return "--"

    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return "--"

    if not isinstance(checkpoint, dict):
        return "--"

    metrics = checkpoint.get("train_metrics")
    if not isinstance(metrics, dict):
        return "--"

    value = metrics.get("metrics/mAP50(M)")
    if value in (None, ""):
        value = metrics.get("metrics/mAP50(B)")
    return format_map50(value)


def model_map50(path: Path | str) -> str:
    checkpoint_value = checkpoint_map50(path)
    if checkpoint_value != "--":
        return checkpoint_value

    results_path = candidate_results_csv(Path(path))
    if not results_path or not results_path.exists():
        return "--"

    try:
        with results_path.open("r", encoding="utf-8", newline="") as file:
            rows = [row for row in csv.DictReader(file) if row]
    except (OSError, csv.Error):
        return "--"

    if not rows:
        return "--"

    latest = rows[-1]
    return format_map50(latest.get("metrics/mAP50(M)") or latest.get("metrics/mAP50(B)"))


def sync_accelerator() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def benchmark_fps(detector: YOLO, warmup: int = 1, repeats: int = 3) -> str:
    try:
        for _ in range(warmup):
            detector.predict(BENCHMARK_IMAGE, verbose=False)
        sync_accelerator()

        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            detector.predict(BENCHMARK_IMAGE, verbose=False)
            sync_accelerator()
            durations.append(time.perf_counter() - start)
    except Exception:
        return "--"

    if not durations:
        return "--"

    average = sum(durations) / len(durations)
    if average <= 0:
        return "--"
    return f"{1 / average:.1f}"


def model_fps(path: Path | str, detector: YOLO | None = None, force: bool = False) -> str:
    key = resolved_key(path)
    if not force and key in MODEL_FPS_CACHE:
        return MODEL_FPS_CACHE[key]

    measured = benchmark_fps(detector or get_model())
    MODEL_FPS_CACHE[key] = measured
    return measured


def model_metrics(path: Path | str, detector: YOLO | None = None) -> dict[str, str]:
    return {
        "map50": model_map50(path),
        "fps": model_fps(path, detector),
    }


def model_display_name(path: Path) -> str:
    name = path.name
    if len(name) > 33 and name[32] == "_":
        prefix = name[:32]
        if all(char in "0123456789abcdef" for char in prefix.lower()):
            return name[33:]
    return name


def available_models() -> list[dict[str, Any]]:
    candidates: list[Path] = []
    if DEFAULT_MODEL.exists():
        candidates.append(DEFAULT_MODEL)
    candidates.extend(
        sorted(
            (path for path in MODEL_DIR.rglob("*") if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    )

    seen: set[str] = set()
    items = []
    active = str(model_path.resolve()).lower()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved.lower() in seen:
            continue
        seen.add(resolved.lower())
        items.append(
            {
                "path": resolved,
                "name": model_display_name(candidate),
                "active": resolved.lower() == active,
            }
        )
    return items


def save_upload(file_storage, prefix: str) -> Path:
    suffix = Path(file_storage.filename or "").suffix.lower()
    safe_suffix = suffix if suffix else ".bin"
    target = UPLOAD_DIR / f"{prefix}_{uuid.uuid4().hex}{safe_suffix}"
    file_storage.save(target)
    return target


def image_from_request() -> tuple[np.ndarray, str]:
    if "image" in request.files:
        uploaded = request.files["image"]
        data = uploaded.read()
        name = uploaded.filename or "camera-frame.jpg"
    else:
        payload = request.get_json(silent=True) or {}
        data_url = payload.get("image", "")
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        data = base64.b64decode(data_url)
        name = payload.get("name") or "camera-frame.jpg"

    array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取图片数据")
    return image, name


def encode_jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("图片编码失败")
    return buffer.tobytes()


def severity_from_area(area_ratio: float, count: int) -> str:
    if count <= 0:
        return "未检测"
    if area_ratio < 3:
        return "轻微"
    if area_ratio < 10:
        return "中等"
    return "严重"


def report_maintenance_sections(counts: dict[str, int] | None) -> list[dict[str, Any]]:
    if not counts:
        return []
    return [
        REPORT_MAINTENANCE_GUIDES[key]
        for key in ("crevice", "pitting", "uniform")
        if int(counts.get(key, 0) or 0) > 0
    ]


def load_report_data(result_id: str) -> dict[str, Any] | None:
    data = reports.get(result_id)
    if data:
        return data

    manifest = OUTPUT_DIR / f"{result_id}.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    return None


def report_compare_item(data: dict[str, Any], inline_images: bool = False) -> dict[str, Any]:
    item = data
    if not data.get("originalUrl") and data.get("items"):
        item = data["items"][0]
    if item.get("originalImage") or item.get("resultImage"):
        item = dict(item)
        item["originalUrl"] = item.get("originalUrl") or item.get("originalImage", "")
        item["resultUrl"] = item.get("resultUrl") or item.get("resultImage", "")
    if not inline_images:
        return item

    exported = dict(item)
    for key in ("originalUrl", "resultUrl"):
        exported[key] = image_url_to_data_uri(str(exported.get(key, "")))
    return exported


def percent_of(part: Any, whole: Any) -> float:
    return round(float(part) / float(whole) * 100, 2) if whole else 0.0


def format_duration(seconds: Any) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        value = 0.0
    if value >= 60:
        minutes = int(value // 60)
        remaining = int(round(value % 60))
        return f"{minutes:02d}:{remaining:02d}"
    return f"{round(value, 1)} s"


def distribution_types(counts: dict[str, Any], total_defects: int) -> list[dict[str, Any]]:
    return [
        {
            "label": "缝隙腐蚀",
            "count": int(counts.get("crevice", 0) or 0),
            "percent": percent_of(counts.get("crevice", 0) or 0, total_defects),
        },
        {
            "label": "点蚀",
            "count": int(counts.get("pitting", 0) or 0),
            "percent": percent_of(counts.get("pitting", 0) or 0, total_defects),
        },
        {
            "label": "均匀腐蚀",
            "count": int(counts.get("uniform", 0) or 0),
            "percent": percent_of(counts.get("uniform", 0) or 0, total_defects),
        },
    ]


def report_batch_analysis(data: dict[str, Any]) -> dict[str, Any]:
    items = data.get("items") or []
    if not items:
        return {"isBatch": False}

    total_images = len(items)
    detected_images = sum(1 for item in items if int(item.get("count", 0) or 0) > 0)
    counts = data.get("counts") or {}
    total_defects = int(data.get("count", 0) or 0)
    severity_counts = {"严重": 0, "中等": 0, "轻微": 0, "未检测": 0}

    for item in items:
        label = "未检测" if int(item.get("count", 0) or 0) <= 0 else str(item.get("severity") or "未检测")
        if label not in severity_counts:
            label = "未检测"
        severity_counts[label] += 1

    return {
        "isBatch": True,
        "summary": {
            "totalImages": total_images,
            "detectedImages": detected_images,
            "undetectedImages": total_images - detected_images,
            "detectedImageRatio": f"{detected_images} / {total_images - detected_images}",
            "totalDefects": total_defects,
            "averageAreaRatio": data.get("areaRatio", 0),
            "averageConfidence": data.get("averageConfidence", 0),
            "elapsedMs": data.get("elapsedMs", 0),
        },
        "types": distribution_types(counts, total_defects),
        "severity": [
            {"label": label, "count": count, "percent": percent_of(count, total_images)}
            for label, count in severity_counts.items()
        ],
    }


def report_video_analysis(data: dict[str, Any]) -> dict[str, Any]:
    frames = data.get("frameMetrics") or []
    if not frames:
        return {"isVideo": False}

    total_frames = len(frames)
    detected_frames = sum(1 for frame in frames if int(frame.get("count", 0) or 0) > 0)
    counts = data.get("counts") or {}
    total_defects = int(data.get("count", 0) or 0)
    severity_counts = {"严重": 0, "中等": 0, "轻微": 0, "未检测": 0}

    for frame in frames:
        label = "未检测" if int(frame.get("count", 0) or 0) <= 0 else str(frame.get("severity") or "未检测")
        if label not in severity_counts:
            label = "未检测"
        severity_counts[label] += 1

    duration_sec = data.get("durationSec")
    if duration_sec is None:
        duration_sec = max((float(frame.get("timeSec", 0) or 0) for frame in frames), default=0)

    return {
        "isVideo": True,
        "summary": {
            "duration": format_duration(duration_sec),
            "detectedFrameRatio": f"{detected_frames} / {total_frames - detected_frames}",
            "totalDefects": total_defects,
            "averageAreaRatio": data.get("areaRatio", 0),
            "averageConfidence": data.get("averageConfidence", 0),
            "elapsedMs": data.get("elapsedMs", 0),
        },
        "types": distribution_types(counts, total_defects),
        "severity": [
            {"label": label, "count": count, "percent": percent_of(count, total_frames)}
            for label, count in severity_counts.items()
        ],
    }


def image_url_to_data_uri(url: str) -> str:
    if not url.startswith("/outputs/"):
        return url

    image_path = OUTPUT_DIR / Path(url).name
    if not image_path.exists() or not image_path.is_file():
        return url

    suffix = image_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
    }.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_report_document(data: dict[str, Any], *, word_export: bool = False) -> str:
    rows = data.get("rows", [])[:80]
    compare_item = report_compare_item(data, inline_images=word_export)
    batch_analysis = report_batch_analysis(data)
    video_analysis = report_video_analysis(data)
    show_detail_rows = not batch_analysis.get("isBatch") and not video_analysis.get("isVideo")
    maintenance_sections = report_maintenance_sections(data.get("counts"))
    return render_template_string(
        REPORT_TEMPLATE,
        data=data,
        rows=rows,
        compare_item=compare_item,
        batch_analysis=batch_analysis,
        video_analysis=video_analysis,
        show_detail_rows=show_detail_rows,
        maintenance_sections=maintenance_sections,
        word_export=word_export,
    )


def summarize_result(result, image_shape: tuple[int, int, int], elapsed_ms: int, result_id: str, source_name: str) -> dict[str, Any]:
    height, width = image_shape[:2]
    image_area = max(height * width, 1)
    boxes = result.boxes
    masks = result.masks

    rows: list[dict[str, Any]] = []
    counts = {"crevice": 0, "pitting": 0, "uniform": 0}
    confidences: list[float] = []
    total_area = 0.0

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        mask_areas: list[float] = []
        if masks is not None and masks.data is not None:
            mask_data = masks.data.cpu().numpy()
            mask_areas = [float(mask.sum()) for mask in mask_data]

        for index, (box, class_id, score) in enumerate(zip(xyxy, cls, conf), start=1):
            x1, y1, x2, y2 = [float(v) for v in box]
            if index - 1 < len(mask_areas):
                area = mask_areas[index - 1]
            else:
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            key = CLASS_KEYS.get(int(class_id), f"class_{class_id}")
            counts[key] = counts.get(key, 0) + 1
            confidences.append(float(score))
            total_area += area
            rows.append(
                {
                    "index": index,
                    "classId": int(class_id),
                    "className": CLASS_NAMES.get(int(class_id), f"类别{class_id}"),
                    "classEn": CLASS_EN.get(int(class_id), f"class{class_id}"),
                    "confidence": round(float(score) * 100, 2),
                    "areaRatio": round(area / image_area * 100, 2),
                    "box": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                    },
                    "position": f"({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})",
                }
            )

    count = len(rows)
    area_ratio = round(total_area / image_area * 100, 2)
    severity = severity_from_area(area_ratio, count)
    average_conf = round(float(np.mean(confidences)) * 100, 2) if confidences else 0
    active_types = sum(1 for value in counts.values() if value > 0)

    return {
        "id": result_id,
        "sourceName": source_name,
        "count": count,
        "averageConfidence": average_conf,
        "typeCount": active_types,
        "elapsedMs": elapsed_ms,
        "areaRatio": area_ratio,
        "severity": severity,
        "counts": counts,
        "rows": rows,
    }


def detect_array(image: np.ndarray, source_name: str, conf: float, iou: float, save_output: bool = True) -> dict[str, Any]:
    detector = get_model()
    result_id = uuid.uuid4().hex
    start = time.perf_counter()
    results = detector.predict(image, conf=conf, iou=iou, verbose=False)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    result = results[0]
    annotated = result.plot()
    summary = summarize_result(result, image.shape, elapsed_ms, result_id, source_name)

    if save_output:
        original_path = OUTPUT_DIR / f"{result_id}_original.jpg"
        result_path = OUTPUT_DIR / f"{result_id}_result.jpg"
        original_path.write_bytes(encode_jpeg(image))
        result_path.write_bytes(encode_jpeg(annotated))
        summary["originalUrl"] = f"/outputs/{original_path.name}"
        summary["resultUrl"] = f"/outputs/{result_path.name}"
        summary["downloadUrl"] = f"/api/download/result/{result_id}"
    else:
        summary["originalImage"] = "data:image/jpeg;base64," + base64.b64encode(encode_jpeg(image, 85)).decode("ascii")
        summary["resultImage"] = "data:image/jpeg;base64," + base64.b64encode(encode_jpeg(annotated, 85)).decode("ascii")

    summary["reportUrl"] = f"/api/report/{result_id}"
    reports[result_id] = summary
    if save_output:
        (OUTPUT_DIR / f"{result_id}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def aggregate_video_summaries(
    frame_summaries: list[dict[str, Any]],
    elapsed_ms: int,
    result_id: str,
    source_name: str,
) -> dict[str, Any]:
    counts = {"crevice": 0, "pitting": 0, "uniform": 0}
    rows: list[dict[str, Any]] = []
    confidences: list[float] = []
    total_count = 0
    total_area_ratio = 0.0

    for frame_index, item in enumerate(frame_summaries, start=1):
        total_count += int(item.get("count", 0))
        total_area_ratio += float(item.get("areaRatio", 0))
        for key in counts:
            counts[key] += int((item.get("counts") or {}).get(key, 0))
        for row in item.get("rows", []):
            if len(rows) < 80:
                video_row = dict(row)
                video_row["frame"] = frame_index
                rows.append(video_row)
            confidences.append(float(row.get("confidence", 0)))

    frame_count = len(frame_summaries)
    area_ratio = round(total_area_ratio / max(frame_count, 1), 2)
    average_conf = round(float(np.mean(confidences)), 2) if confidences else 0
    return {
        "id": result_id,
        "sourceName": source_name,
        "count": total_count,
        "averageConfidence": average_conf,
        "typeCount": sum(1 for value in counts.values() if value > 0),
        "elapsedMs": elapsed_ms,
        "areaRatio": area_ratio,
        "severity": severity_from_area(area_ratio, total_count),
        "counts": counts,
        "rows": rows,
        "frameCount": frame_count,
    }


def video_frame_metric(frame_summary: dict[str, Any], frame_index: int, fps: float) -> dict[str, Any]:
    rows = [dict(row) for row in frame_summary.get("rows", [])[:20]]
    return {
        "frameIndex": frame_index,
        "timeSec": round((frame_index - 1) / max(float(fps), 1.0), 3),
        "count": int(frame_summary.get("count", 0)),
        "averageConfidence": frame_summary.get("averageConfidence", 0),
        "typeCount": frame_summary.get("typeCount", 0),
        "elapsedMs": frame_summary.get("elapsedMs", 0),
        "areaRatio": frame_summary.get("areaRatio", 0),
        "severity": frame_summary.get("severity", "未检测"),
        "counts": frame_summary.get("counts", {"crevice": 0, "pitting": 0, "uniform": 0}),
        "rows": rows,
    }


def create_video_writer(path: Path, fourcc: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    if not writer.isOpened():
        writer.release()
        raise ValueError(f"无法创建视频结果文件: {path.name}")
    return writer


def detect_video_file(video_path: Path, source_name: str, conf: float, iou: float) -> dict[str, Any]:
    detector = get_model()
    result_id = uuid.uuid4().hex
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("无法读取视频尺寸")

    result_path = OUTPUT_DIR / f"{result_id}_result.webm"
    writer = create_video_writer(result_path, "VP80", fps, (width, height))
    frame_summaries: list[dict[str, Any]] = []
    frame_metrics: list[dict[str, Any]] = []
    frame_index = 0
    last_annotated: np.ndarray | None = None
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            should_detect = (frame_index - 1) % VIDEO_DETECT_STRIDE == 0
            if should_detect:
                frame_start = time.perf_counter()
                results = detector.predict(frame, conf=conf, iou=iou, verbose=False)
                elapsed_ms = int((time.perf_counter() - frame_start) * 1000)
                result = results[0]
                last_annotated = result.plot()
                frame_summary = summarize_result(result, frame.shape, elapsed_ms, result_id, source_name)
                frame_summaries.append(frame_summary)
                frame_metrics.append(video_frame_metric(frame_summary, frame_index, fps))
            writer.write(last_annotated if last_annotated is not None else frame)
    finally:
        cap.release()
        writer.release()

    if not frame_summaries:
        raise ValueError("视频中没有可处理的帧")

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    summary = aggregate_video_summaries(frame_summaries, elapsed_ms, result_id, source_name)
    summary["frameCount"] = frame_index
    summary["frameMetrics"] = frame_metrics
    summary["detectedFrameCount"] = len(frame_summaries)
    summary["durationSec"] = round(frame_index / max(float(fps), 1.0), 3)
    summary["resultVideoUrl"] = f"/outputs/{result_path.name}"
    summary["downloadUrl"] = f"/api/download/result/{result_id}"
    summary["reportUrl"] = f"/api/report/{result_id}"
    reports[result_id] = summary
    (OUTPUT_DIR / f"{result_id}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/outputs/<path:filename>")
def output_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


@app.get("/api/model")
def api_model():
    try:
        detector = get_model()
        metrics = model_metrics(model_path, detector)
        loaded = True
        error = ""
    except Exception as exc:
        loaded = False
        error = str(exc)
        metrics = {"map50": "--", "fps": "--"}
    return jsonify(
        {
            "connected": True,
            "loaded": loaded,
            "error": error,
            "modelPath": str(model_path),
            "modelName": model_display_name(Path(model_path)),
            "defaultModel": str(DEFAULT_MODEL),
            "models": available_models(),
            "classes": [{"id": key, "name": CLASS_NAMES[key], "en": CLASS_EN[key]} for key in sorted(CLASS_NAMES)],
            "map50": metrics["map50"],
            "fps": metrics["fps"],
        }
    )


@app.post("/api/model")
def api_set_model():
    if "modelPath" in request.form or (request.is_json and (request.get_json(silent=True) or {}).get("modelPath")):
        payload = request.get_json(silent=True) or {}
        selected_path = request.form.get("modelPath") or payload.get("modelPath")
        target = Path(selected_path or "")
        if target.suffix.lower() not in MODEL_SUFFIXES:
            return jsonify({"ok": False, "error": "仅支持 .pt / .onnx / .engine 模型文件"}), 400
        allowed = {item["path"].lower() for item in available_models()}
        if str(target.resolve()).lower() not in allowed:
            return jsonify({"ok": False, "error": "模型不在历史导入列表中"}), 400
        try:
            detector = load_model(target)
            metrics = {"map50": model_map50(model_path), "fps": model_fps(model_path, detector, force=True)}
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify(
            {
                "ok": True,
                "modelPath": str(model_path),
                "modelName": model_display_name(Path(model_path)),
                "models": available_models(),
                "map50": metrics["map50"],
                "fps": metrics["fps"],
            }
        )

    if "model" not in request.files:
        return jsonify({"ok": False, "error": "请上传模型文件"}), 400
    uploaded = request.files["model"]
    original_name = Path(uploaded.filename or "model").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in MODEL_SUFFIXES:
        return jsonify({"ok": False, "error": "仅支持 .pt / .onnx / .engine 模型文件"}), 400
    target = MODEL_DIR / f"{uuid.uuid4().hex}_{original_name}"
    uploaded.save(target)
    try:
        detector = load_model(target)
        metrics = {"map50": model_map50(model_path), "fps": model_fps(model_path, detector, force=True)}
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(
        {
            "ok": True,
            "modelPath": str(model_path),
            "modelName": model_display_name(Path(model_path)),
            "models": available_models(),
            "map50": metrics["map50"],
            "fps": metrics["fps"],
        }
    )


@app.post("/api/detect/image")
def api_detect_image():
    try:
        conf = float(request.form.get("conf", 0.45))
        iou = float(request.form.get("iou", 0.5))
        image, name = image_from_request()
        summary = detect_array(image, name, conf, iou)
        return jsonify({"ok": True, "result": summary})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/detect/batch")
def api_detect_batch():
    try:
        conf = float(request.form.get("conf", 0.45))
        iou = float(request.form.get("iou", 0.5))
        files = request.files.getlist("images")
        if not files:
            return jsonify({"ok": False, "error": "请选择图片文件夹"}), 400

        batch_id = uuid.uuid4().hex
        batch_dir = OUTPUT_DIR / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        results = []
        total_counts = {"crevice": 0, "pitting": 0, "uniform": 0}
        total_count = 0
        total_area = 0.0
        confidences = []

        for uploaded in files:
            data = uploaded.read()
            array = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None:
                continue
            item = detect_array(image, uploaded.filename or "batch-image.jpg", conf, iou)
            src_name = Path(item["resultUrl"]).name
            shutil.copyfile(OUTPUT_DIR / src_name, batch_dir / src_name)
            results.append(item)
            total_count += item["count"]
            total_area += item["areaRatio"]
            confidences.extend(row["confidence"] for row in item["rows"])
            for key in total_counts:
                total_counts[key] += int(item["counts"].get(key, 0))

        average_conf = round(float(np.mean(confidences)), 2) if confidences else 0
        area_ratio = round(total_area / max(len(results), 1), 2)
        severity = severity_from_area(area_ratio, total_count)
        summary = {
            "id": batch_id,
            "sourceName": f"批量检测 {len(results)} 张",
            "count": total_count,
            "averageConfidence": average_conf,
            "typeCount": sum(1 for value in total_counts.values() if value > 0),
            "elapsedMs": sum(item["elapsedMs"] for item in results),
            "areaRatio": area_ratio,
            "severity": severity,
            "counts": total_counts,
            "rows": [row for item in results for row in item["rows"]],
            "items": results,
            "downloadUrl": f"/api/download/result/{batch_id}",
            "reportUrl": f"/api/report/{batch_id}",
        }
        reports[batch_id] = summary
        (OUTPUT_DIR / f"{batch_id}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify({"ok": True, "result": summary})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/detect/batch/finalize")
def api_finalize_batch():
    try:
        payload = request.get_json(silent=True) or {}
        ids = payload.get("ids") or []
        items = [reports[result_id] for result_id in ids if result_id in reports]
        if not items:
            return jsonify({"ok": False, "error": "没有可汇总的检测结果"}), 400

        batch_id = uuid.uuid4().hex
        batch_dir = OUTPUT_DIR / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        total_counts = {"crevice": 0, "pitting": 0, "uniform": 0}
        confidences = []
        total_count = 0
        total_area = 0.0
        rows = []

        for item in items:
            result_url = item.get("resultUrl", "")
            if result_url:
                src_name = Path(result_url).name
                src_path = OUTPUT_DIR / src_name
                if src_path.exists():
                    shutil.copyfile(src_path, batch_dir / src_name)
            total_count += int(item.get("count", 0))
            total_area += float(item.get("areaRatio", 0))
            rows.extend(item.get("rows", []))
            confidences.extend(row.get("confidence", 0) for row in item.get("rows", []))
            for key in total_counts:
                total_counts[key] += int((item.get("counts") or {}).get(key, 0))

        average_conf = round(float(np.mean(confidences)), 2) if confidences else 0
        area_ratio = round(total_area / max(len(items), 1), 2)
        severity = severity_from_area(area_ratio, total_count)
        summary = {
            "id": batch_id,
            "sourceName": f"批量检测 {len(items)} 张",
            "count": total_count,
            "averageConfidence": average_conf,
            "typeCount": sum(1 for value in total_counts.values() if value > 0),
            "elapsedMs": sum(int(item.get("elapsedMs", 0)) for item in items),
            "areaRatio": area_ratio,
            "severity": severity,
            "counts": total_counts,
            "rows": rows,
            "items": items,
            "downloadUrl": f"/api/download/result/{batch_id}",
            "reportUrl": f"/api/report/{batch_id}",
        }
        reports[batch_id] = summary
        (OUTPUT_DIR / f"{batch_id}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify({"ok": True, "result": summary})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/detect/frame")
def api_detect_frame():
    try:
        payload = request.get_json(silent=True) or {}
        conf = float(payload.get("conf", 0.45))
        iou = float(payload.get("iou", 0.5))
        image, name = image_from_request()
        summary = detect_array(image, name, conf, iou, save_output=False)
        return jsonify({"ok": True, "result": summary})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/detect/video")
def api_detect_video():
    try:
        if "video" not in request.files:
            return jsonify({"ok": False, "error": "请选择视频文件"}), 400
        conf = float(request.form.get("conf", 0.45))
        iou = float(request.form.get("iou", 0.5))
        uploaded = request.files["video"]
        source_name = uploaded.filename or "inspection-video.mp4"
        video_path = save_upload(uploaded, "video")
        summary = detect_video_file(video_path, source_name, conf, iou)
        return jsonify({"ok": True, "result": summary})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/download/result/<result_id>")
def api_download(result_id: str):
    batch_dir = OUTPUT_DIR / result_id
    if batch_dir.exists() and batch_dir.is_dir():
        zip_path = OUTPUT_DIR / f"{result_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in batch_dir.iterdir():
                if file.is_file():
                    archive.write(file, file.name)
        return send_file(zip_path, as_attachment=True, download_name=f"corrosion_result_{result_id}.zip")

    video_file = OUTPUT_DIR / f"{result_id}_result.mp4"
    if video_file.exists():
        return send_file(video_file, as_attachment=True, download_name=f"corrosion_result_{result_id}.mp4", mimetype="video/mp4")

    result_file = OUTPUT_DIR / f"{result_id}_result.jpg"
    if result_file.exists():
        return send_file(result_file, as_attachment=True, download_name=f"corrosion_result_{result_id}.jpg")
    return jsonify({"ok": False, "error": "结果不存在"}), 404


REPORT_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>工业腐蚀检测报告</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #eef4f5;
      color: #102033;
      font-family: "Microsoft YaHei", Arial, sans-serif;
    }
    .page {
      width: min(1180px, calc(100% - 48px));
      margin: 28px auto;
      padding: 30px;
      border: 1px solid #dbe4ed;
      border-radius: 12px;
      background: #fff;
    }
    .report-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 24px;
      padding-bottom: 22px;
      border-bottom: 1px solid #dbe4ed;
    }
    .report-title {
      min-width: 0;
      flex: 1 1 auto;
    }
    h1 { margin: 0 0 10px; font-size: 28px; line-height: 1.2; }
    h2 { margin: 26px 0 14px; font-size: 20px; }
    .meta {
      color: #66758a;
      line-height: 1.8;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .report-actions {
      flex: 0 0 auto;
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: flex-end;
      padding-right: 0;
      max-width: 100%;
    }
    .export-button {
      display: inline-grid;
      place-items: center;
      width: 124px;
      height: 44px;
      border: 0;
      border-radius: 8px;
      padding: 0;
      appearance: none;
      color: #fff;
      background: #0d8178;
      font-family: "Microsoft YaHei", Arial, sans-serif;
      font-size: 16px;
      font-weight: 700;
      line-height: 1;
      cursor: pointer;
      text-decoration: none;
      box-shadow: 0 8px 18px rgba(13, 129, 120, .22);
    }
    .batch-panel {
      display: grid;
      gap: 14px;
      margin-top: 12px;
    }
    .single-summary,
    .batch-summary {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 22px;
    }
    .metric-card {
      border: 1px solid #dbe4ed;
      border-radius: 8px;
      padding: 14px;
      background: #fbfdfd;
    }
    .metric-card span {
      display: block;
      color: #66758a;
      font-size: 13px;
    }
    .metric-card strong {
      display: block;
      margin-top: 6px;
      color: #102033;
      font-size: 22px;
    }
    .distribution-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .distribution-card {
      border: 1px solid #dbe4ed;
      border-radius: 8px;
      padding: 16px;
      background: #fbfdfd;
    }
    .distribution-card h3 {
      margin: 0 0 14px;
      color: #102033;
      font-size: 17px;
    }
    .distribution-row {
      display: grid;
      grid-template-columns: 86px 1fr 72px;
      gap: 10px;
      align-items: center;
      margin: 12px 0;
      color: #43566d;
      font-size: 14px;
    }
    .bar-track {
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: #e4edf2;
    }
    .bar-fill {
      height: 100%;
      border-radius: 999px;
      background: #0d8178;
    }
    .distribution-value {
      text-align: right;
      color: #102033;
      font-weight: 700;
    }
    .compare-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    .image-card {
      border: 1px solid #dbe4ed;
      border-radius: 10px;
      overflow: hidden;
      background: #f8fbfd;
    }
    .image-title {
      padding: 12px 14px;
      border-bottom: 1px solid #dbe4ed;
      color: #43566d;
      font-weight: 700;
      background: #f2f7f9;
    }
    .image-frame {
      height: 430px;
      display: grid;
      place-items: center;
      padding: 12px;
    }
    .image-frame img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      border: 0;
      border-radius: 6px;
    }
    .suggestions {
      margin: 0;
      display: grid;
      gap: 14px;
    }
    .maintenance-card {
      border: 1px solid #dbe4ed;
      border-radius: 8px;
      background: #fbfdfd;
      color: #43566d;
      overflow: hidden;
    }
    .maintenance-head {
      padding: 16px 18px;
      border-bottom: 1px solid #dbe4ed;
      background: #f3faf8;
    }
    .maintenance-head h3 {
      margin: 0 0 10px;
      color: #102033;
      font-size: 18px;
    }
    .pain-point {
      margin: 0;
      padding: 10px 12px;
      border-left: 4px solid #0d8178;
      border-radius: 6px;
      background: #e8f6f3;
      color: #315d59;
      line-height: 1.6;
    }
    .maintenance-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      padding: 16px;
    }
    .maintenance-group {
      border: 1px solid #e1e9f0;
      border-radius: 8px;
      padding: 14px 14px 12px;
      background: #fff;
    }
    .maintenance-group h4 {
      margin: 0 0 10px;
      color: #21354a;
      font-size: 15px;
    }
    .maintenance-group ul {
      margin: 0;
      padding-left: 19px;
    }
    .maintenance-group li {
      margin: 7px 0;
      line-height: 1.65;
    }
    table {
      width: 100%;
      border: 1px solid #dbe4ed;
      border-radius: 8px;
      border-collapse: separate;
      border-spacing: 0;
      overflow: hidden;
    }
    th, td {
      border-bottom: 1px solid #dbe4ed;
      padding: 11px 12px;
      text-align: left;
      font-size: 14px;
      white-space: nowrap;
    }
    th { background: #f3f7fa; color: #54677d; }
    tr:last-child td { border-bottom: 0; }
    @media (max-width: 820px) {
      .page { width: calc(100% - 24px); padding: 18px; }
      .report-header { flex-direction: column; }
      .single-summary, .batch-summary, .distribution-grid, .compare-grid, .maintenance-grid { grid-template-columns: 1fr; }
      .report-actions { width: 100%; flex-wrap: wrap; justify-content: stretch; }
      .export-button { flex: 1 1 150px; }
      .image-frame { height: 320px; }
    }
    @media print {
      body { background: #fff; }
      .page { width: 100%; margin: 0; padding: 16px; border: 0; border-radius: 0; }
      .report-actions { display: none; }
      .single-summary,
      .batch-summary { grid-template-columns: repeat(3, 1fr); }
      .distribution-grid { grid-template-columns: 1fr 1fr; gap: 10px; }
      .compare-grid { grid-template-columns: 1fr 1fr; gap: 10px; }
      .maintenance-grid { grid-template-columns: 1fr 1fr; gap: 8px; padding: 10px; }
      .distribution-card, .maintenance-card, .maintenance-group { break-inside: avoid; }
      .image-frame { height: 300px; }
      h2 { margin-top: 18px; }
    }
  </style>
</head>
<body>
  <main class="page">
    <header class="report-header">
      <div class="report-title">
        <h1>YOLO11 工业腐蚀检测报告</h1>
        <div class="meta">
          来源：{{ data.sourceName }}<br>
          编号：{{ data.id }}
        </div>
      </div>
      {% if not word_export %}
      <div class="report-actions">
        <button class="export-button" onclick="window.print()">导出 PDF</button>
        <a class="export-button" href="/api/report/{{ data.id }}/word">导出 Word</a>
      </div>
      {% endif %}
    </header>

    {% if show_detail_rows %}
    <section class="single-summary">
      <div class="metric-card"><span>检测目标</span><strong>{{ data.count }}</strong></div>
      <div class="metric-card"><span>平均置信度</span><strong>{{ data.averageConfidence }}%</strong></div>
      <div class="metric-card"><span>面积占比</span><strong>{{ data.areaRatio }}%</strong></div>
      <div class="metric-card"><span>腐蚀类型</span><strong>{{ data.typeCount }}类</strong></div>
      <div class="metric-card"><span>处理耗时</span><strong>{{ data.elapsedMs }} ms</strong></div>
      <div class="metric-card"><span>严重程度</span><strong>{{ data.severity }}</strong></div>
    </section>
    {% endif %}

    {% if batch_analysis.isBatch %}
    <h2>批量统计摘要</h2>
    <section class="batch-panel">
      <div class="batch-summary">
        <div class="metric-card"><span>总图片数</span><strong>{{ batch_analysis.summary.totalImages }}</strong></div>
        <div class="metric-card"><span>检出 / 未检出图片数</span><strong>{{ batch_analysis.summary.detectedImageRatio }}</strong></div>
        <div class="metric-card"><span>总缺陷数量</span><strong>{{ batch_analysis.summary.totalDefects }}</strong></div>
        <div class="metric-card"><span>平均面积占比</span><strong>{{ batch_analysis.summary.averageAreaRatio }}%</strong></div>
        <div class="metric-card"><span>平均置信度</span><strong>{{ batch_analysis.summary.averageConfidence }}%</strong></div>
        <div class="metric-card"><span>处理耗时</span><strong>{{ batch_analysis.summary.elapsedMs }} ms</strong></div>
      </div>
      <div class="distribution-grid">
        <article class="distribution-card">
          <h3>腐蚀类型分布</h3>
          {% for item in batch_analysis.types %}
          <div class="distribution-row">
            <span>{{ item.label }}</span>
            <div class="bar-track"><div class="bar-fill" style="width: {{ item.percent }}%;"></div></div>
            <span class="distribution-value">{{ item.percent }}%</span>
          </div>
          {% endfor %}
        </article>
        <article class="distribution-card">
          <h3>风险等级分布</h3>
          {% for item in batch_analysis.severity %}
          <div class="distribution-row">
            <span>{{ item.label }}</span>
            <div class="bar-track"><div class="bar-fill" style="width: {{ item.percent }}%;"></div></div>
            <span class="distribution-value">{{ item.percent }}%</span>
          </div>
          {% endfor %}
        </article>
      </div>
    </section>
    {% elif video_analysis.isVideo %}
    <h2>视频统计摘要</h2>
    <section class="batch-panel">
      <div class="batch-summary">
        <div class="metric-card"><span>视频时长</span><strong>{{ video_analysis.summary.duration }}</strong></div>
        <div class="metric-card"><span>检出 / 未检出帧数</span><strong>{{ video_analysis.summary.detectedFrameRatio }}</strong></div>
        <div class="metric-card"><span>总缺陷数量</span><strong>{{ video_analysis.summary.totalDefects }}</strong></div>
        <div class="metric-card"><span>平均面积占比</span><strong>{{ video_analysis.summary.averageAreaRatio }}%</strong></div>
        <div class="metric-card"><span>平均置信度</span><strong>{{ video_analysis.summary.averageConfidence }}%</strong></div>
        <div class="metric-card"><span>处理耗时</span><strong>{{ video_analysis.summary.elapsedMs }} ms</strong></div>
      </div>
      <div class="distribution-grid">
        <article class="distribution-card">
          <h3>腐蚀类型分布</h3>
          {% for item in video_analysis.types %}
          <div class="distribution-row">
            <span>{{ item.label }}</span>
            <div class="bar-track"><div class="bar-fill" style="width: {{ item.percent }}%;"></div></div>
            <span class="distribution-value">{{ item.percent }}%</span>
          </div>
          {% endfor %}
        </article>
        <article class="distribution-card">
          <h3>风险等级分布</h3>
          {% for item in video_analysis.severity %}
          <div class="distribution-row">
            <span>{{ item.label }}</span>
            <div class="bar-track"><div class="bar-fill" style="width: {{ item.percent }}%;"></div></div>
            <span class="distribution-value">{{ item.percent }}%</span>
          </div>
          {% endfor %}
        </article>
      </div>
    </section>
    {% elif compare_item.originalUrl or compare_item.resultUrl %}
    <h2>图像对比</h2>
    <section class="compare-grid">
      <article class="image-card">
        <div class="image-title">原始图像</div>
        <div class="image-frame">
          {% if compare_item.originalUrl %}<img src="{{ compare_item.originalUrl }}" alt="原始图像">{% else %}<span>暂无原图</span>{% endif %}
        </div>
      </article>
      <article class="image-card">
        <div class="image-title">检测结果</div>
        <div class="image-frame">
          {% if compare_item.resultUrl %}<img src="{{ compare_item.resultUrl }}" alt="检测结果">{% else %}<span>暂无检测图</span>{% endif %}
        </div>
      </article>
    </section>
    {% endif %}

    {% if maintenance_sections %}
    <h2>维护建议</h2>
    <section class="suggestions">
      {% for section in maintenance_sections %}
      <article class="maintenance-card">
        <div class="maintenance-head">
          <h3>{{ section.title }}</h3>
          <p class="pain-point"><strong>核心痛点：</strong>{{ section.pain }}</p>
        </div>
        <div class="maintenance-grid">
          {% for group in section.groups %}
          <div class="maintenance-group">
            <h4>{{ group.title }}</h4>
            <ul>
              {% for item in group["items"] %}
              <li>{{ item }}</li>
              {% endfor %}
            </ul>
          </div>
          {% endfor %}
        </div>
      </article>
      {% endfor %}
    </section>
    {% endif %}

    {% if show_detail_rows %}
    <h2>检测明细</h2>
    <table>
      <thead><tr><th>类别</th><th>置信度</th><th>面积占比</th><th>位置</th></tr></thead>
      <tbody>
        {% for row in rows %}
        <tr><td>{{ row.className }}</td><td>{{ row.confidence }}%</td><td>{{ row.areaRatio }}%</td><td>{{ row.position }}</td></tr>
        {% else %}
        <tr><td colspan="4">暂无检测结果</td></tr>
        {% endfor %}
      </tbody>
    </table>
    {% endif %}
  </main>
</body>
</html>
"""


@app.get("/api/report/<result_id>")
def api_report(result_id: str):
    data = load_report_data(result_id)
    if not data:
        return jsonify({"ok": False, "error": "报告不存在或服务已重启"}), 404
    return render_report_document(data)


@app.get("/api/report/<result_id>/word")
def api_report_word(result_id: str):
    data = load_report_data(result_id)
    if not data:
        return jsonify({"ok": False, "error": "报告不存在或服务已重启"}), 404

    html = "\ufeff" + render_report_document(data, word_export=True)
    response = Response(html, content_type="application/msword; charset=utf-8")
    response.headers["Content-Disposition"] = f'attachment; filename="corrosion_report_{result_id}.doc"'
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
