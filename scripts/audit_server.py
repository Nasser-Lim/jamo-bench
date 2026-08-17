# -*- coding: utf-8 -*-
"""사람 감사(§9, target-blind transcription) 로컬 웹 UI 서버.

Judge Ceiling 측정에서 CLOVA가 봤던 것과 동일한 clean 렌더링을
결정론적으로 재생성해 사람에게 target-blind로 보여주고, 무슨 글자로
읽었는지 받는다. 목적은 셀별 사람 정확도 vs CLOVA 정확도를 비교해
"진짜 안 읽히는 글자"와 "CLOVA만의 사전 편향"을 가려내는 것 —
지난 Ceiling 실측(18셀 전부 90% 미만, 특히 겹받침 셀 0%)이 CLOVA의
한계인지 실제 판독 불가능인지 확인하는 게 이 세션의 목표.

실행:
    python scripts/audit_server.py
브라우저에서 http://127.0.0.1:8877 접속.

응답은 results/audit/human_audit.jsonl에 감사자별로 누적 저장되며,
같은 감사자 이름으로 다시 접속하면 이미 답한 문항은 건너뛴다(재개 가능).
서로 다른 감사자는 독립적으로 전체 큐를 처음부터 판독한다(§9.2 —
2인 이상의 독립 판정이 있어야 Krippendorff's α를 계산할 수 있다).
"""
from __future__ import annotations

import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jamo_bench.audit_queue import CANVAS_SIZE, TEXT_AREA_FRAC, AuditItem, build_audit_queue  # noqa: E402
from jamo_bench.forge_render import render_clean  # noqa: E402

HOST = "127.0.0.1"
PORT = 8877
UI_HTML_PATH = Path(__file__).resolve().parent / "audit_ui.html"
RESULTS_PATH = REPO_ROOT / "results" / "audit" / "human_audit.jsonl"
DISPLAY_SIZE = 512  # 전송 크기 축소(내용은 CLOVA가 본 것과 동일, 해상도만 축소)

_lock = threading.Lock()
_queue: List[AuditItem] = build_audit_queue()
_items_by_id: Dict[str, AuditItem] = {item.item_id: item for item in _queue}
_image_cache: Dict[Tuple[str, str], str] = {}  # (char, font_name) -> data URI


def _render_data_uri(char: str, font_name: str) -> str:
    key = (char, font_name)
    if key in _image_cache:
        return _image_cache[key]
    img = render_clean(char, font_name=font_name, canvas_size=CANVAS_SIZE, text_area_frac=TEXT_AREA_FRAC)
    img = img.resize((DISPLAY_SIZE, DISPLAY_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    _image_cache[key] = uri
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


def _load_all_records() -> List[dict]:
    if not RESULTS_PATH.is_file():
        return []
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 콘솔 스팸 억제
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
                    "cell_key": item.cell_key,
                    "font_name": item.font_name,
                    "image_data_uri": _render_data_uri(item.char, item.font_name),
                },
            })
            return

        if parsed.path == "/api/summary":
            annotator = parse_qs(parsed.query).get("annotator", ["anonymous"])[0]
            with _lock:
                answered = _load_answered(annotator)
            self._send_json(_compute_summary(annotator, answered))
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

        human_correct = label == "text" and transcription == item.char
        agree_with_clova = (
            label == "text"
            and item.clova_reading is not None
            and transcription == item.clova_reading
        )

        record = {
            "id": item_id,
            "annotator": annotator,
            "cell_key": item.cell_key,
            "font_name": item.font_name,
            "target": item.char,  # 서버 로그에는 정답을 남긴다 — 클라이언트에만 숨겼다
            "clova_reading": item.clova_reading,
            "clova_correct": item.clova_correct,
            "label": label,
            "transcription": transcription,
            "human_correct": human_correct,
            "agree_with_clova": agree_with_clova,
        }

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with RESULTS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._send_json({"ok": True})


def _compute_summary(annotator: str, answered: Dict[str, dict]) -> dict:
    records = list(answered.values())
    total = len(records)
    if total == 0:
        return {"total": 0, "human_accuracy": 0, "clova_accuracy": 0, "agreement_rate": 0, "per_cell": []}

    human_acc = sum(1 for r in records if r["human_correct"]) / total
    clova_acc = sum(1 for r in records if r["clova_correct"]) / total
    agree = sum(1 for r in records if r["agree_with_clova"]) / total

    by_cell: Dict[str, List[dict]] = {}
    for r in records:
        by_cell.setdefault(r["cell_key"], []).append(r)

    per_cell = []
    for cell_key, recs in by_cell.items():
        n = len(recs)
        per_cell.append({
            "cell_key": cell_key,
            "n": n,
            "human_accuracy": sum(1 for r in recs if r["human_correct"]) / n,
            "clova_accuracy": sum(1 for r in recs if r["clova_correct"]) / n,
        })
    per_cell.sort(key=lambda c: c["human_accuracy"])

    return {
        "total": total,
        "human_accuracy": human_acc,
        "clova_accuracy": clova_acc,
        "agreement_rate": agree,
        "per_cell": per_cell,
    }


def main():
    print(f"문항 {len(_queue)}개 로드됨 (judge_ceiling.json 기준, clean 조건만)")
    print(f"http://{HOST}:{PORT} 에서 접속하세요")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
