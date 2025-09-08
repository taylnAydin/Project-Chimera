# -*- coding: utf-8 -*-
"""
Notifier Agent
- Reporter'dan gelen raporu (title, body, path, symbol, run_id) kanallara iletir.
- Eğer reporter zaten diske yazdıysa tekrar yazmaz; sadece konumu bildirir.
- path yoksa symbol/run_id ile güvenli bir dosya adı oluşturup kaydeder.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import os
from datetime import datetime

def _ensure_reports_dir() -> str:
    path = os.path.join("data", "reports")
    os.makedirs(path, exist_ok=True)
    return path

def _safe_filename(symbol: Optional[str], run_id: Optional[str]) -> str:
    sym = (symbol or "UNKNOWN").upper()
    rid = run_id or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{sym}_run_{rid}.md"

def send(report: Dict[str, Any], cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Args:
        report: reporter.build(...) çıktısı (title, body, path, symbol, run_id içerir)
        cfg: kanal ayarları (şimdilik kullanılmıyor)

    Behavior:
        - path varsa tekrar yazmaz; sadece bildirir.
        - path yoksa data/reports altında symbol/run_id ile kaydeder.
    """
    cfg = cfg or {}
    title  = report.get("title") or "Chimera Raporu"
    body   = report.get("body") or ""
    path   = report.get("path")  # reporter zaten yazdıysa dolu olur
    symbol = report.get("symbol") or cfg.get("symbol")
    run_id = report.get("run_id") or cfg.get("run_id")

    banner = "=" * 80
    print(banner)
    print(f"[Chimera Notifier] {title}")
    print(banner)

    if path and os.path.isfile(path):
        # Reporter yazmış → sadece duyur
        print(f"[notifier] Rapor zaten kaydedildi: {path}\n")
        return {"ok": True, "path": path, "symbol": symbol, "run_id": run_id}

    # Reporter path vermemişse biz kaydedelim
    reports_dir = _ensure_reports_dir()
    filename = _safe_filename(symbol, run_id)
    out_path = os.path.join(reports_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"[notifier] Rapor kaydedildi: {out_path}\n")
    return {"ok": True, "path": out_path, "symbol": symbol, "run_id": run_id}