# -*- coding: utf-8 -*-
"""사람 감사(§9) Phase 2 — 파일럿 300장 라벨 완결 + 라벨 체계 개편.

`audit_server_pilot.py`(포트 8878)가 만든 1차 감사 256건은
`route == "B" and jamo_valid == "VALID"`로 걸러진 집합이었다. 그 결과:

  1. Route A1/C·OVERGEN 이미지 44장을 사람이 한 번도 본 적이 없다 —
     "Route 분류기가 무효 출력을 걸러내는가"에 답할 수 없는 상태.
  2. `illegible` 44건 안에 malformed(획이 틀린 한글)·multi_syllable·
     non_hangul이 섞여 있는데 라벨이 하나로 뭉쳐 있다.

이 서버는 둘을 한 번에 해소한다(합계 87문항). 라벨 체계는
`audit_queue.PHASE2_LABELS` 5종이며, 감사자가 1차에서 실제로 적용했던 기준
("윈도우 키보드로 입력이 불가능한 형태")을 화면 문구로 명문화한다 — 2번째
감사자가 같은 기준을 적용할 수 있어야 Krippendorff α가 의미를 갖는다.

**1차 결과 파일은 건드리지 않는다.** 결과는 별도 파일에 쓰고, 분석 단계에서
`PHASE1_LABEL_MAP`으로 합친다.

실행:
    python scripts/audit_server_pilot2.py
브라우저에서 http://127.0.0.1:8879 (8877 Ceiling, 8878 1차와 겹치지 않는 포트)

결과: results/audit/human_audit_pilot_phase2.jsonl
"""
from __future__ import annotations

import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Windows 기본 콘솔은 cp949라 한글·em dash 출력에서 죽는다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jamo_bench.audit_queue import (  # noqa: E402
    PHASE2_LABELS,
    PilotAuditItem,
    build_pilot_phase2_queue,
)

HOST = "127.0.0.1"
PORT = 8879
UI_HTML_PATH = Path(__file__).resolve().parent / "audit_ui_pilot2.html"
RESULTS_PATH = REPO_ROOT / "results" / "audit" / "human_audit_pilot_phase2.jsonl"
PILOT_JSONL = REPO_ROOT / "results" / "pilot" / "pilot_results_v2.jsonl"

_MIME_BY_SUFFIX = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

_lock = threading.Lock()
_queue: List[PilotAuditItem] = build_pilot_phase2_queue()
_items_by_id: Dict[str, PilotAuditItem] = {item.item_id: item for item in _queue}
_image_cache: Dict[str, str] = {}

# 감사자에게는 숨기지만 기록에는 남긴다 — Route precision 계산의 핵심 축.
_meta_by_path: Dict[str, dict] = {
    r["image_path"]: r
    for r in (
        json.loads(line)
        for line in PILOT_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
}

_TRANSCRIPTION_REQUIRED = {label for label, _, needs in PHASE2_LABELS if needs}


def _image_data_uri(image_path: str) -> str:
    if image_path in _image_cache:
        return _image_cache[image_path]
    path = REPO_ROOT / image_path
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    uri = f"data:{mime};base64,{b64}"
    _image_cache[image_path] = uri
    return uri


def _load_answered(annotator: str) -> Dict[str, dict]:
    answered: Dict[str, dict] = {}
    if not RESULTS_PATH.is_file():
        return answered
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("annotator") == annotator:
                answered[rec["id"]] = rec
    return answered


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = UI_HTML_PATH.read_text(encoding="utf-8").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parsed.path == "/api/labels":
            self._send_json({
                "labels": [
                    {"value": v, "text": t, "needs_transcription": n}
                    for v, t, n in PHASE2_LABELS
                ]
            })
            return

        if parsed.path == "/api/next":
            annotator = parse_qs(parsed.query).get("annotator", ["anonymous"])[0]
            with _lock:
                answered = _load_answered(annotator)
                remaining = [it for it in _queue if it.item_id not in answered]
            if not remaining:
                self._send_json({"done": True})
                return
            item = remaining[0]
            self._send_json({
                "done": False,
                "index": len(_queue) - len(remaining) + 1,
                "total": len(_queue),
                "item": {
                    "id": item.item_id,
                    "group": item.item_id.split("::", 1)[0],
                    "image_data_uri": _image_data_uri(item.image_path),
                },
            })
            return

        if parsed.path == "/api/summary":
            annotator = parse_qs(parsed.query).get("annotator", ["anonymous"])[0]
            with _lock:
                answered = _load_answered(annotator)
            self._send_json(_compute_summary(answered))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/api/submit":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))

        item = _items_by_id.get(body.get("id"))
        if item is None:
            self._send_json({"ok": False, "error": "unknown item id"}, status=400)
            return

        label = body.get("label")
        if label not in {v for v, _, _ in PHASE2_LABELS}:
            self._send_json({"ok": False, "error": f"unknown label: {label}"}, status=400)
            return

        transcription = (body.get("transcription") or "").strip()
        if label in _TRANSCRIPTION_REQUIRED and not transcription:
            self._send_json({"ok": False, "error": "transcription required"}, status=400)
            return
        if label not in _TRANSCRIPTION_REQUIRED:
            transcription = ""

        meta = _meta_by_path.get(item.image_path, {})
        record = {
            "id": item.item_id,
            "group": item.item_id.split("::", 1)[0],
            "annotator": body.get("annotator") or "anonymous",
            "image_path": item.image_path,
            "target": item.target,
            "template_id": item.template_id,
            "coda_type": item.coda_type,
            # Route precision 계산용 — 감사자에게는 보이지 않았다.
            "route": meta.get("route"),
            "jamo_valid": meta.get("jamo_valid"),
            "label": label,
            "transcription": transcription,
            "note": (body.get("note") or "").strip(),
            "human_correct": label == "valid_syllable" and transcription == item.target,
        }

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with RESULTS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._send_json({"ok": True})


def _compute_summary(answered: Dict[str, dict]) -> dict:
    records = list(answered.values())
    if not records:
        return {"total": 0, "by_label": [], "route_crosstab": []}

    counts: Dict[str, int] = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    by_label = [
        {"label": v, "text": t, "n": counts.get(v, 0)} for v, t, _ in PHASE2_LABELS
    ]

    # 미감사 집단만 — 이게 이번 감사의 산출물이다. Route가 무효 출력을
    # 실제로 걸러내고 있었는지는 이 표로만 답할 수 있다.
    rows: Dict[str, Dict[str, int]] = {}
    for r in records:
        if r["group"] != "unaudited":
            continue
        route = r.get("route") or "?"
        bucket = rows.setdefault(route, {"n": 0, "valid_syllable": 0, "invalid": 0})
        bucket["n"] += 1
        if r["label"] == "valid_syllable":
            bucket["valid_syllable"] += 1
        else:
            bucket["invalid"] += 1

    crosstab = [
        {"route": route, **vals, "invalid_rate": vals["invalid"] / vals["n"]}
        for route, vals in sorted(rows.items())
    ]
    return {"total": len(records), "by_label": by_label, "route_crosstab": crosstab}


def main():
    n_unaudited = sum(1 for it in _queue if it.item_id.startswith("unaudited::"))
    print(f"문항 {len(_queue)}개 "
          f"(미감사 {n_unaudited} + 재라벨링 {len(_queue) - n_unaudited})")
    print("미감사 집단이 먼저 나옵니다 — Route A1/C precision 측정에 직결됩니다.")
    print(f"http://{HOST}:{PORT} 에서 접속하세요")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
