# agents/notifier.py
"""
Chimera • Notifier Agent
- Reporter tarafından üretilen raporu kullanıcıya ulaştırır.
- Şu anda iki şey yapıyor:
  1) Konsola yazdırır.
  2) data/reports/ klasörüne .md dosyası olarak kaydeder.
"""

from __future__ import annotations
import os
from typing import Dict, Any
from datetime import datetime

def _ensure_reports_dir(path: str = "data/reports") -> str:
    """data/reports klasörü yoksa oluşturur, yolunu döner."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def send(report: Dict[str, Any], cfg: Dict[str, Any] | None = None) -> None:
    """
    Raporu konsola yaz ve data/reports klasörüne kaydet.

    Args:
        report: reporter.build çıktısı. Beklenen alanlar:
            - report["title"]: str
            - report["body"]: str (Markdown içeriği)
            - report["meta"]: dict (symbol, run_id, vs.)
        cfg: opsiyonel config (şimdilik kullanılmıyor).
    """
    if not report:
        print("[notifier] Uyarı: boş rapor alındı.")
        return

    title = report.get("title", "Chimera Raporu")
    body  = report.get("body", "")
    meta  = report.get("meta", {})

    # 1) Konsola yaz
    print("="*80)
    print(f"[Chimera Notifier] {title}")
    print("="*80)
    print(body)
    print("="*80)

    # 2) Dosyaya kaydet
    reports_dir = _ensure_reports_dir("data/reports")
    symbol = meta.get("symbol", "UNKNOWN")
    run_id = meta.get("run_id", datetime.utcnow().strftime("%Y%m%d%H%M%S"))
    filename = f"{symbol}_run_{run_id}.md"

    filepath = os.path.join(reports_dir, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"[notifier] Rapor kaydedildi: {filepath}")
    except Exception as e:
        print(f"[notifier] Rapor dosyaya yazılamadı: {e}")