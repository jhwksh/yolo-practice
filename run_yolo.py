"""사진 한 장을 YOLO detection / segmentation 으로 돌려 conf 0.25 와 0.50 결과를 비교한다.

사용법:  python run_yolo.py input/bus.jpg
결과:    outputs/yolo/<사진이름>/ 아래에 결과 이미지, 박스 JSON, 마스크 PNG, manifest.json
"""
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

IMGSZ = 640
DEVICE = "cpu"
CONFS = [0.25, 0.50, 0.70]
MODELS = {"detect": "yolo26n.pt", "segment": "yolo26n-seg.pt"}


def box_records(result):
    names = result.names
    records = []
    for i, b in enumerate(result.boxes):
        rec = {
            "class_id": int(b.cls),
            "class_name": names[int(b.cls)],
            "confidence": round(float(b.conf), 4),
            "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
        }
        if result.masks is not None:
            rec["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in result.masks.xy[i]]
        records.append(rec)
    return records


def save_mask_png(result, path):
    h, w = result.orig_shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    if result.masks is not None:
        for poly in result.masks.xy:
            if len(poly):
                cv2.fillPoly(canvas, [poly.astype(np.int32)], 255)
    cv2.imwrite(str(path), canvas)


def main(image_path):
    image = Path(image_path)
    out_dir = Path("outputs") / "yolo" / image.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for task, weights in MODELS.items():
        model = YOLO(weights)
        for conf in CONFS:
            result = model.predict(str(image), conf=conf, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
            tag = f"{task}_conf{conf:.2f}"
            cv2.imwrite(str(out_dir / f"{tag}.jpg"), result.plot())
            records = box_records(result)
            (out_dir / f"{tag}.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if task == "segment":
                save_mask_png(result, out_dir / f"{tag}_mask.png")
            summary.append({"task": task, "weights": weights, "conf": conf, "num_detections": len(records)})
            print(f"{tag}: {len(records)}개")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(image),
        "imgsz": IMGSZ,
        "device": DEVICE,
        "run_location": "로컬 PC (Windows, CPU만 사용, GPU 없음)",
        "confs": CONFS,
        "models": MODELS,
        "versions": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
        },
        "runs": summary,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장 위치:", out_dir.resolve())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("사용법: python run_yolo.py <이미지 경로>")
    main(sys.argv[1])
