# -*- coding: utf-8 -*-
"""사람 감사(§9) 로컬 웹 UI — 실제 Seedream 파일럿 이미지용.

`audit_server.py`(Judge Ceiling의 폰트 렌더링용)와 달리 이 서버는
**실제 생성 이미지**를 디스크에서 읽어 보여준다. 폰트 렌더링 감사는
template_match의 템플릿과 같은 폰트라 순환 검증이 되므로, template_match를
제대로 검증하려면 실제 T2I 출력이 필요하다.

한 번의 사람 판독으로 세 가지를 동시에 얻는다:
  1. 사람 vs 실제 타깃           → Seedream의 진짜 정확도(그동안 못 구했던 답)
  2. 사람 vs CLOVA(occ 0.10)     → CLOVA가 실제 생성물에서 신뢰할 만한가
  3. 사람 vs template_match      → template_match가 실제 생성물에서 신뢰할 만한가

문항은 CLOVA와 template_match가 서로 다르게 읽은 것부터 배치한다(가장
정보가치 높은 지점 우선, §9.1과 같은 원칙).

실행:
    python scripts/audit_server_pilot.py
브라우저에서 http://127.0.0.1:8878 접속 (Ceiling용 audit_server.py의
8877과 겹치지 않는 포트 — 두 감사를 동시에 열어둘 수 있다).

결과: results/audit/human_audit_pilot.jsonl
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

from jamo_bench.audit_queue import PilotAuditItem, build_pilot_audit_queue  # noqa: E402

HOST = "127.0.0.1"
PORT = 8878
UI_HTML_PATH = Path(__file__).resolve().parent / "audit_ui_pilot.html"
RESULTS_PATH = REPO_ROOT / "results" / "audit" / "human_audit_pilot.jsonl"

_MIME_BY_SUFFIX = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

_lock = threading.Lock()
_queue: List[PilotAuditItem] = build_pilot_audit_queue()
_items_by_id: Dict[str, PilotAuditItem] = {item.item_id: item for item in _queue}
_image_cache: Dict[str, str] = {}


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

        if parsed.path == "/api/next":
            annotator = parse_qs(parsed.query).get("annotator", ["anonymous"])[0]
            with _lock:
                answered = _load_answered(annotator)
                remaining = [it for it in _queue if it.item_id not in answered]
            if not remaining:
                self._send_json({"done": True})
                return
            item = remaining[0]
            index = len(_queue) - len(remaining) + 1
            self._send_json({
                "done": False,
                "index": index,
                "total": len(_queue),
                "item": {
                    "id": item.item_id,
                    "coda_type": item.coda_type,
                    "template_id": item.template_id,
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

        item_id = body.get("id")
        item = _items_by_id.get(item_id)
        if item is None:
            self._send_json({"ok": False, "error": "unknown item id"}, status=400)
            return

        label = body.get("label", "text")
        transcription = (body.get("transcription") or "").strip()
        annotator = body.get("annotator") or "anonymous"
        is_text = label == "text"

        record = {
            "id": item_id,
            "annotator": annotator,
            "coda_type": item.coda_type,
            "template_id": item.template_id,
            "image_path": item.image_path,
            "target": item.target,
            "clova_reading": item.clova_reading,
            "clova_correct": item.clova_correct,
            "template_match_reading": item.template_match_reading,
            "template_match_correct": item.template_match_correct,
            "label": label,
            "transcription": transcription,
            "human_correct": is_text and transcription == item.target,
            "agree_with_clova": is_text and item.clova_reading is not None and transcription == item.clova_reading,
            "agree_with_tm": is_text
            and item.template_match_reading is not None
            and transcription == item.template_match_reading,
        }

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with RESULTS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._send_json({"ok": True})


def _compute_summary(answered: Dict[str, dict]) -> dict:
    records = list(answered.values())
    total = len(records)
    if total == 0:
        return {
            "total": 0, "human_accuracy": 0, "clova_accuracy": 0, "tm_accuracy": 0,
            "clova_agreement_rate": 0, "tm_agreement_rate": 0, "clova_vs_tm_agreement": 0,
            "per_cell": [],
        }

    human_acc = sum(r["human_correct"] for r in records) / total
    clova_acc = sum(r["clova_correct"] for r in records) / total
    tm_acc = sum(r["template_match_correct"] for r in records) / total
    clova_agree = sum(r["agree_with_clova"] for r in records) / total
    tm_agree = sum(r["agree_with_tm"] for r in records) / total
    clova_vs_tm = sum(1 for r in records if r["clova_reading"] == r["template_match_reading"]) / total

    by_cell: Dict[str, List[dict]] = {}
    for r in records:
        by_cell.setdefault(r["coda_type"], []).append(r)

    per_cell = []
    for cell_key, recs in by_cell.items():
        n = len(recs)
        per_cell.append({
            "cell_key": cell_key,
            "n": n,
            "human_accuracy": sum(rr["human_correct"] for rr in recs) / n,
            "clova_accuracy": sum(rr["clova_correct"] for rr in recs) / n,
            "tm_accuracy": sum(rr["template_match_correct"] for rr in recs) / n,
        })
    per_cell.sort(key=lambda c: c["human_accuracy"])

    return {
        "total": total,
        "human_accuracy": human_acc,
        "clova_accuracy": clova_acc,
        "tm_accuracy": tm_acc,
        "clova_agreement_rate": clova_agree,
        "tm_agreement_rate": tm_agree,
        "clova_vs_tm_agreement": clova_vs_tm,
        "per_cell": per_cell,
    }


def main():
    n_disagree = sum(1 for it in _queue if it.clova_reading != it.template_match_reading)
    print(f"문항 {len(_queue)}개 로드됨 (CLOVA↔template_match 불일치 {n_disagree}건이 먼저 나옵니다)")
    print(f"http://{HOST}:{PORT} 에서 접속하세요")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
