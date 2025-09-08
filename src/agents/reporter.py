# agents/reporter.py
"""
Chimera • Reporter
- Pipeline artefaktlarını (quality, best, regime, risk_plan, anomalies, mode)
  okunabilir bir RAPORA dönüştürür.
- LLM (Gemini) ile doğal dil "narrative" üretir; başarısızlıkta deterministik fallback kullanır.
- .env ve cfg ile yapılandırılabilir.

Env örneği:
  GEMINI_API_KEY=...
  GEMINI_MODEL=gemini-1.5-pro
  GEMINI_TIMEOUT=15
  REPORTER_USE_LLM=true
  REPORTER_MAX_TOKENS=800

cfg örneği (router.run'da geç):
  cfg = {
    "report": {
      "use_llm": True,
      "max_tokens": 800,
      "timeout_s": 15,
      "model_order": ["gemini-1.5-pro", "gemini-1.5-flash"],  # deneme sırası
      "redact_before_llm": True
    }
  }
"""

from __future__ import annotations
import os
import time
from typing import TypedDict, Literal, Dict, Any, Optional, List
from datetime import datetime

# --- Opsiyonel: dotenv varsa .env'yi yükle (yoksa sessiz geç)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --- Gemini SDK (pip install google-generativeai)
try:
    import google.generativeai as genai  # type: ignore
except Exception as _e:
    genai = None  # type: ignore

# ==========================
# Tipler
# ==========================

class ReportMeta(TypedDict, total=False):
    symbol: str
    run_id: str
    mode: Literal["full", "analyze", "block"]
    coverage_class: Optional[str]
    coverage_years: Optional[float]
    rmse: Optional[float]
    aic: Optional[float]
    regime_trend: Optional[str]
    regime_vol: Optional[str]
    risk_position: Optional[float]
    created_at: str

class Report(TypedDict, total=False):
    format: Literal["markdown", "html", "text"]
    title: str
    body: str   # markdown varsayılan
    meta: ReportMeta

# ==========================
# Env / Varsayılanlar
# ==========================

_ENV_API_KEY = os.getenv("GEMINI_API_KEY")
_ENV_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
_ENV_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "15"))
_ENV_USE_LLM = os.getenv("REPORTER_USE_LLM", "true").lower() == "true"
_ENV_MAX_TOK = int(os.getenv("REPORTER_MAX_TOKENS", "800"))

_DEFAULT_MODEL_ORDER = [_ENV_MODEL, "gemini-1.5-flash"]  # deneme önceliği


# ==========================
# Redaksiyon (opsiyonel)
# ==========================

def _redact(obj: Any) -> Any:
    """
    LLM'e gitmeden önce hassas alanları maskelemek için basit redaksiyon.
    İstersen burada PII maskerini entegre edebilirsin (telefon/email/iban vs).
    Şimdilik sadece sayısal olmayan uzun stringleri olduğu gibi bırakıyor.
    """
    # Basit örnek: hiç dokunma; placeholder. Geliştirilebilir.
    return obj


# ==========================
# Prompt & Fallback
# ==========================

def _compose_prompt(symbol: str,
                    best: Dict[str, Any] | None,
                    regime: Dict[str, Any] | None,
                    risk_plan: Dict[str, Any] | None,
                    quality: Dict[str, Any] | None,
                    anomalies: List[Dict[str, Any]] | None,
                    mode: str) -> str:
    """LLM'e gidecek kısa ve yapılandırılmış Türkçe prompt."""
    lines = [
        "Görev: Aşağıdaki trading pipeline çıktılarından kısa, açık ve teknik bir Türkçe özet üret.",
        "Abartılı dil kullanma; rakamları ve varsayımları net ver.",
        f"Sembol: {symbol}",
        f"Mod: {mode}",
    ]
    if quality:
        lines.append(f"Veri kalitesi: kapsam_yıl={quality.get('coverage_years')} sınıf={quality.get('coverage_class')}")
        lines.append(f"NaN%={quality.get('nan_pct')} Dup%={quality.get('duplicate_pct')} Gap%={quality.get('gap_pct')} Outlier%={quality.get('outlier_ratio')}")
    if best:
        lines.append(f"Model: {best.get('name')} | RMSE={best.get('rmse')} | AIC={best.get('aic')}")
    if regime:
        lines.append(f"Rejim: trend={regime.get('trend')} vol={regime.get('vol_class')}")
    if risk_plan and mode == "full":
        lines.append(f"Risk: pozisyon={risk_plan.get('position_size')} SL={risk_plan.get('stop_loss')} TP={risk_plan.get('take_profit')}")
    if anomalies:
        lines.append(f"Anomali_sayısı={len(anomalies)}")
    lines.append("Yanıtı Markdown başlıkları ve maddeleriyle ver. Gerekiyorsa 'Uyarılar' ve 'Varsayımlar' bölümü ekle.")
    return "\n".join(lines)


def _fallback_markdown(symbol, best, regime, risk_plan, quality, anomalies, mode) -> str:
    """LLM kapalı/başarısızsa deterministik rapor üret."""
    lines = [
        f"# Chimera Raporu — {symbol}",
        f"- Mod: **{mode}**",
    ]
    if quality:
        lines += [
            "## Veri Kalitesi",
            f"- Kapsam (yıl): {quality.get('coverage_years')}",
            f"- Sınıf: {quality.get('coverage_class')}",
            f"- NaN%: {quality.get('nan_pct')}",
            f"- Dup%: {quality.get('duplicate_pct')}",
            f"- Gap%: {quality.get('gap_pct')}",
            f"- Outlier%: {quality.get('outlier_ratio')}",
        ]
    if best:
        lines += [
            "## Model",
            f"- En iyi model: {best.get('name')}",
            f"- RMSE: {best.get('rmse')}, AIC: {best.get('aic')}",
        ]
    if regime:
        lines += [
            "## Rejim",
            f"- Trend: {regime.get('trend')}",
            f"- Volatilite: {regime.get('vol_class')}",
        ]
    if risk_plan and mode == "full":
        lines += [
            "## Risk Planı",
            f"- Pozisyon: {risk_plan.get('position_size')}",
            f"- Stop Loss: {risk_plan.get('stop_loss')}",
            f"- Take Profit: {risk_plan.get('take_profit')}",
        ]
    if anomalies:
        lines += [
            "## Anomaliler",
            f"- Tespit edilen anomali sayısı: {len(anomalies)}"
        ]
    lines += [
        "## Notlar",
        "- Bu rapor LLM kapalı/erişilemezken **şablon** ile oluşturulmuştur.",
        "- `cfg.report.use_llm=true` ve `.env` ile LLM özetini açabilirsin."
    ]
    return "\n".join(lines)


# ==========================
# Gemini Çağrısı (backoff + fallback)
# ==========================

class _GeminiError(RuntimeError):
    pass

def _ensure_gemini(api_key: Optional[str]) -> None:
    if genai is None:
        raise _GeminiError("google-generativeai paketini kur: `pip install google-generativeai`")
    if not api_key:
        raise _GeminiError("GEMINI_API_KEY bulunamadı (.env veya ortam değişkeni).")

def _call_gemini_with_backoff(prompt: str,
                              model_order: List[str],
                              api_key: str,
                              max_tokens: int,
                              timeout_s: int) -> str:
    """
    model_order'daki modelleri sırayla dene (ör. ['gemini-1.5-pro','gemini-1.5-flash']).
    429 gibi hatalarda kısa backoff yapıp bir sonraki modeli dener.
    Başarısızsa _GeminiError fırlatır.
    """
    _ensure_gemini(api_key)
    genai.configure(api_key=api_key)

    # Basit backoff aralıkları (saniye)
    backoffs = [0, 10, 30]

    last_err: Optional[Exception] = None
    for model_name in model_order:
        for wait in backoffs:
            if wait:
                time.sleep(wait)
            try:
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": max_tokens},
                    safety_settings=None,  # istersen özelleştir
                    request_options={"timeout": timeout_s}
                )
                text = getattr(resp, "text", None) or ""
                if not text.strip():
                    raise _GeminiError("Gemini boş yanıt döndü.")
                return text
            except Exception as e:
                last_err = e
                # 429/ResourceExhausted ise backoff döngüsü devam edecek; yoksa direkt diğer modele geç
                if not (("ResourceExhausted" in str(e)) or ("429" in str(e))):
                    break  # bu modelde farklı bir hata; sıradaki modele geç
        # bir sonraki model adıyla denemeye devam
    raise _GeminiError(f"Gemini çağrısı başarısız: {last_err}")


# ==========================
# Ana Giriş: build(...)
# ==========================

def build(run_id: str,
          symbol: str,
          best: Dict[str, Any] | None,
          regime: Dict[str, Any] | None,
          risk_plan: Dict[str, Any] | None,
          quality: Dict[str, Any] | None,
          anomalies: List[Dict[str, Any]] | None,
          mode: str,
          cfg: Dict[str, Any] | None) -> Report:
    """
    Router çağırır. LLM kullanılacaksa Gemini ile özet üretir; aksi halde fallback markdown döner.

    Args:
      run_id: Bu run'ın kimliği (router/persistence üretir)
      symbol: Örn. 'BTCUSDT'
      best: model_selection çıktısı (name, rmse, aic, ...)
      regime: regime_detector çıktısı (trend, vol_class, ...)
      risk_plan: risk.plan çıktısı (position_size, stop_loss, take_profit, ...)
      quality: data_quality.evaluate çıktısı
      anomalies: anomaly.detect listesi
      mode: 'full' | 'analyze' | 'block'
      cfg: reporter konfigürasyonu (use_llm, max_tokens, model_order, timeout_s, redact_before_llm)

    Returns:
      Report: { format, title, body, meta }
    """
    cfg = cfg or {}
    use_llm: bool = bool(cfg.get("use_llm", _ENV_USE_LLM))
    max_tokens: int = int(cfg.get("max_tokens", _ENV_MAX_TOK))
    timeout_s: int = int(cfg.get("timeout_s", _ENV_TIMEOUT))
    model_order: List[str] = list(cfg.get("model_order", _DEFAULT_MODEL_ORDER))
    redact: bool = bool(cfg.get("redact_before_llm", True))
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    # Meta
    meta: ReportMeta = {
        "symbol": symbol,
        "run_id": run_id,
        "mode": mode,
        "coverage_class": quality.get("coverage_class") if quality else None,
        "coverage_years": quality.get("coverage_years") if quality else None,
        "rmse": (best or {}).get("rmse"),
        "aic": (best or {}).get("aic"),
        "regime_trend": (regime or {}).get("trend"),
        "regime_vol": (regime or {}).get("vol_class"),
        "risk_position": (risk_plan or {}).get("position_size") if risk_plan else None,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    title = f"Chimera Raporu • {symbol} • run {run_id}"

    # LLM kapalı veya API key yoksa → fallback
    if not use_llm or not api_key:
        body = _fallback_markdown(symbol, best, regime, risk_plan, quality, anomalies, mode)
        return Report(format="markdown", title=title, body=body, meta=meta)

    # LLM açık → prompt hazırla (opsiyonel redaksiyon)
    _best = _redact(best) if redact else best
    _regime = _redact(regime) if redact else regime
    _risk = _redact(risk_plan) if redact else risk_plan
    _quality = _redact(quality) if redact else quality
    _anoms = _redact(anomalies) if redact else anomalies

    prompt = _compose_prompt(symbol, _best, _regime, _risk, _quality, _anoms, mode)

    # Gemini çağrısı; 429 vb. durumlara karşı backoff + model fallback
    try:
        body = _call_gemini_with_backoff(
            prompt=prompt,
            model_order=model_order,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout_s=timeout_s
        )
    except Exception:
        # LLM başarısız → fallback
        body = _fallback_markdown(symbol, best, regime, risk_plan, quality, anomalies, mode)

    return Report(format="markdown", title=title, body=body, meta=meta)