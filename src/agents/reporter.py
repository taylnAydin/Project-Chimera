# -*- coding: utf-8 -*-
"""
Reporter Agent
==============
- Pipeline state'inden insan okunur Markdown rapor üretir.
- Opsiyonel: Gemini API ile metni "insansı" dille parlatır.
- Çıktıyı konsola basar ve data/reports klasörüne kaydeder.

Kullanım (router içinde):
    report = reporter.build(
        run_id=state["run_id"],
        symbol=state["symbol"],
        best=state["best"],
        regime=state["regime"],
        risk_plan=state["risk_plan"],
        quality=state["quality"],
        anomalies=state["anomalies"],
        mode=state["mode"],
        cfg=state["cfg"].get("report", {})
    )
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import textwrap
from datetime import datetime

import pandas as pd

# -----------------------------------------------------------------------------
# Gemini opsiyonel bağlama
# -----------------------------------------------------------------------------

def _maybe_gemini_polish(markdown_text: str, cfg: Dict[str, Any]) -> str:
    """
    REPORTER_USE_LLM=true ve GEMINI_API_KEY varsa teksit parlatmayı dener.
    Hata olursa orijinal metni geri döner.
    """
    use_llm = str(os.getenv("REPORTER_USE_LLM", "false")).lower() == "true" or bool(cfg.get("use_llm"))
    api_key = os.getenv("GEMINI_API_KEY") or cfg.get("gemini_api_key")
    model_name = cfg.get("gemini_model", "gemini-1.5-flash")

    if not (use_llm and api_key):
        return markdown_text

    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        sys_prompt = (
            "Aşağıdaki trading raporunu Türkçe olarak doğal, akıcı ve net bir dille parlat. "
            "Teknik içeriği KORU, sayıları değiştirme. Gereksiz süsleme ve satış dili kullanma."
        )
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content([sys_prompt, markdown_text])
        polished = (resp.text or "").strip()
        if polished:
            return polished
        return markdown_text
    except Exception:
        # LLM başarısızsa sessizce fallback
        return markdown_text

# -----------------------------------------------------------------------------
# Yardımcı formatlayıcılar
# -----------------------------------------------------------------------------

def _fmt_usd(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)

def _fmt_int(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)

def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        return f"{100*float(x):.2f}%"
    except Exception:
        return str(x)

def _fmt_date(d) -> str:
    try:
        return pd.to_datetime(d).date().isoformat()
    except Exception:
        return str(d)

def _ensure_reports_dir() -> str:
    path = os.path.join("data", "reports")
    os.makedirs(path, exist_ok=True)
    return path

# -----------------------------------------------------------------------------
# Ana derleyici
# -----------------------------------------------------------------------------

def _assemble_markdown(
    run_id: str,
    symbol: str,
    mode: str,
    quality: Dict[str, Any],
    best: Dict[str, Any],
    regime: Dict[str, Any],
    risk_plan: Dict[str, Any],
    anomalies: List[Dict[str, Any]],
) -> str:
    # --- Quality özet ---
    qrep = (quality or {}).get("report") or {}
    coverage_years = (quality or {}).get("coverage_years")
    coverage_class = (quality or {}).get("coverage_class")
    nan_ratio_max = (quality or {}).get("nan_ratio_max")
    dup_ratio = (quality or {}).get("duplicate_ratio")
    gap_ratio = (quality or {}).get("gap_ratio")
    zero_vol_ratio = (quality or {}).get("zero_volume_ratio")
    outliers = (quality or {}).get("outlier_count")

    # --- Model / forecast ---
    model_name = (best or {}).get("meta", {}).get("model") or (best or {}).get("name", "")
    model_cfg  = (best or {}).get("meta", {}).get("cfg") or (best or {}).get("meta", {}).get("order")
    horizon    = (best or {}).get("meta", {}).get("horizon") or (best or {}).get("horizon")
    rmse       = (best or {}).get("meta", {}).get("select_rmse") or (best or {}).get("score")
    aic        = (best or {}).get("meta", {}).get("aic")
    yhat: Optional[pd.Series] = (best or {}).get("yhat")

    # Forecast tarih aralığı
    fc_start = _fmt_date(yhat.index[0]) if isinstance(yhat, pd.Series) and len(yhat) else "-"
    fc_end   = _fmt_date(yhat.index[-1]) if isinstance(yhat, pd.Series) and len(yhat) else "-"

    # --- Rejim ---
    trend         = (regime or {}).get("trend")
    trend_slope   = (regime or {}).get("trend_slope")
    vol_class     = (regime or {}).get("vol_class")
    vol_ann       = (regime or {}).get("vol_annualized")
    vol_z         = (regime or {}).get("vol_z")

    # --- Risk ---
    direction     = (risk_plan or {}).get("direction")
    entry_price   = (risk_plan or {}).get("price")
    units         = (risk_plan or {}).get("units")
    notional      = (risk_plan or {}).get("notional")
    risk_pct_eff  = (risk_plan or {}).get("risk_pct_effective")
    stop_price    = (risk_plan or {}).get("stop_price")
    tp_price      = (risk_plan or {}).get("take_profit_price")
    atr           = (risk_plan or {}).get("atr")
    atr_window    = (risk_plan or {}).get("atr_window")

    # --- Uyarılar otomatik üret ---
    warnings_md = []
    if rmse is None and aic is None:
        warnings_md.append("- Model performans metrikleri (RMSE/AIC) raporda eksik görünüyor.")
    if not risk_plan:
        warnings_md.append("- Risk planı oluşturulamadı.")
    if yhat is None or len(yhat) == 0:
        warnings_md.append("- Tahmin serisi bulunamadı.")
    if coverage_years is not None and float(coverage_years) < 1.0:
        warnings_md.append("- Veri kapsamı 1 yılın altında (düşük güven).")

    # --- Nihai karar (çok basit kural) ---
    if mode != "full":
        final_decision = f"Mod `{mode}` olduğundan işlem açma **önerilmez**."
    elif not risk_plan or (units is not None and float(units) <= 0):
        final_decision = "Risk planı uygun değil → **işlem açma**."
    else:
        # direction bilgisi varsa
        if str(direction).lower() == "short":
            final_decision = "Strateji **short** yönünde küçük boyutlu pozisyonu destekliyor."
        else:
            final_decision = "Strateji **long** yönünde küçük boyutlu pozisyonu destekliyor."

    # --- Forecast tablosu ---
    if isinstance(yhat, pd.Series) and len(yhat):
        rows = [
            f"| {_fmt_date(idx)} | {_fmt_usd(val)} |"
            for idx, val in yhat.items()
        ]
        table_md = "\n".join(["| Tarih | Tahmini Fiyat (USDT) |", "|---|---|", *rows])
    else:
        table_md = "_Tahmin bulunamadı._"

    # --- Rapor gövdesi ---
    md = f"""# {symbol} Trading Pipeline Özeti

## Temel Bilgiler
* **Sembol:** {symbol}
* **Mod:** {mode}
* **Veri Kalitesi:** {coverage_years if coverage_years is not None else '-'} yıl (sınıf: {coverage_class or '-'})
* **NaN Maks Oran:** {(_fmt_pct(nan_ratio_max) if nan_ratio_max is not None else '-')}  
* **Duplicate Oranı:** {(_fmt_pct(dup_ratio) if dup_ratio is not None else '-')}  
* **Gap Oranı:** {(_fmt_pct(gap_ratio) if gap_ratio is not None else '-')}  
* **Sıfır Hacim Oranı:** {(_fmt_pct(zero_vol_ratio) if zero_vol_ratio is not None else '-')}  
* **Aykırı Gün Sayısı:** {outliers if outliers is not None else '-'}

## Modelleme
* **Model:** {model_name or '-'}
* **Konfig:** `{model_cfg}`  
* **Horizon:** {horizon if horizon is not None else '-'} gün  
* **Performans:** RMSE = {rmse if rmse is not None else '-'}, AIC = {aic if aic is not None else '-'}

> Tahmin aralığı: **{fc_start} → {fc_end}**

## Piyasa Durumu
* **Trend:** {trend or '-'} (eğim: {(_fmt_int(trend_slope) if trend_slope is not None else '-')})
* **Volatilite:** {vol_class or '-'} (yıllık ≈ {(_fmt_pct(vol_ann) if vol_ann is not None else '-')}, z={vol_z if vol_z is not None else '-'})

## Risk Yönetimi
| Alan | Değer |
|------|-------|
| Yön | {direction or '-'} |
| Giriş Fiyatı | { _fmt_usd(entry_price) } |
| Boyut (units) | {units if units is not None else '-'} |
| Notional (USDT) | { _fmt_usd(notional) } |
| Efektif Risk | { _fmt_pct(risk_pct_eff) } |
| Stop-Loss | { _fmt_usd(stop_price) } |
| Take-Profit | { _fmt_usd(tp_price) } |
| ATR ({atr_window if atr_window is not None else '-'}) | { _fmt_usd(atr) } |

## Uyarılar
{('\n'.join(warnings_md) if warnings_md else '- herhangi bir kritik uyarı yok -')}

## Varsayımlar
* Geçmiş verinin geleceği temsil gücü sınırlıdır.  
* Model, parametreleri değişmedikçe kısa vadede stabil performans gösterir.  

---

## 📌 Nihai Karar
{final_decision}

### 7 Günlük {symbol} Tahminleri
{table_md}

---
_Run id: {run_id} • Rapor zamanı: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC_
"""
    return textwrap.dedent(md).strip()

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build(
    *,
    run_id: str,
    symbol: str,
    best: Dict[str, Any],
    regime: Dict[str, Any],
    risk_plan: Dict[str, Any],
    quality: Dict[str, Any],
    anomalies: List[Dict[str, Any]],
    mode: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Raporu oluşturur, (opsiyonel) LLM ile parlatır, konsola basar ve diske yazar.
    Returns:
        {
          "title": str,
          "body": str,      # markdown
          "path": str,      # dosya
          "meta": { ... }   # özet
        }
    """
    cfg = cfg or {}

    title = f"Chimera Raporu • {symbol} • run {run_id}"
    body_md = _assemble_markdown(run_id, symbol, mode, quality, best, regime, risk_plan, anomalies)
    body_final = _maybe_gemini_polish(body_md, cfg)

    # Konsola banner
    banner = "=" * 80
    print(banner)
    print(f"[Chimera Notifier] {title}")
    print(banner)
    print(body_final[:2000] + ("\n...\n" if len(body_final) > 2000 else ""))  # uzun raporu kısaltarak yaz

    # Dosyaya yaz
    reports_dir = _ensure_reports_dir()
    file_path = os.path.join(reports_dir, f"{symbol}_run_{run_id}.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(body_final)

    print(f"\n[notifier] Rapor kaydedildi: {file_path}\n")

    # Dönen meta
    meta = {
        "forecast_range": {
            "start": (best or {}).get("yhat").index[0].strftime("%Y-%m-%d")
                      if isinstance((best or {}).get("yhat"), pd.Series) and len((best or {}).get("yhat")) else None,
            "end":   (best or {}).get("yhat").index[-1].strftime("%Y-%m-%d")
                      if isinstance((best or {}).get("yhat"), pd.Series) and len((best or {}).get("yhat")) else None,
        },
        "has_llm": str(os.getenv("REPORTER_USE_LLM", "false")).lower() == "true",
    }

    return {"title": title, "body": body_final, "path": file_path, "meta": meta}