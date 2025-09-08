# agents/guardrails_agent.py
"""
Chimera • Guardrails Agent
- Veri kalitesi metriklerini (data_quality) eşiklerle karşılaştırır.
- Pipeline akış modunu belirler: 'full' | 'analyze' | 'block'.
- İlk ağır ihlalde karar verir, ayrıca tüm ihlalleri 'violations' içinde döner.
"""

from __future__ import annotations
from typing import TypedDict, Literal, Dict, Any, List, Optional


# =========================
# Tipler
# =========================

Mode = Literal["full", "analyze", "block"]

class GuardrailsConfig(TypedDict, total=False):
    """
    Guardrails eşikleri. Yüzdeler 0-1 aralığında verilir.
    - *_full: 'full' mod için üst sınırlar / alt sınırlar
    - *_analyze: 'analyze' mod için tolerans; bunu aşınca 'block'
    """
    # Kapsam (yıl)
    min_years_full: float        # >= bu ise 'full' (diğer metrikler de uygunsa)
    min_years_analyze: float     # < bu ise 'block', arası 'analyze'

    # NaN oranı
    max_nan_full: float
    max_nan_analyze: float

    # Duplicate oranı
    max_duplicate_full: float
    max_duplicate_analyze: float

    # Gap (eksik gün/çubuk) oranı
    max_gap_full: float
    max_gap_analyze: float

    # Outlier oranı
    max_outlier_full: float
    max_outlier_analyze: float

    # Davranış bayrakları
    allow_block_on_no_data: bool   # True ise kalite raporu yoksa 'block'
    force_mode: Mode               # Verilen mod ile zorla (test/manuel override)


class GuardDecision(TypedDict, total=False):
    mode: Mode
    reason: Optional[str]
    violations: List[str]


# =========================
# Varsayılan Eşikler (makul, konservatif)
# =========================

DEFAULT_CFG: GuardrailsConfig = {
    # Kapsam: 3+ yıl -> full, 1–3 yıl -> analyze, <1 yıl -> block
    "min_years_full": 3.0,
    "min_years_analyze": 1.0,

    # NaN oranı: full için %2; analyze için %5
    "max_nan_full": 0.02,
    "max_nan_analyze": 0.05,

    # Duplicate: full için %1; analyze için %3
    "max_duplicate_full": 0.01,
    "max_duplicate_analyze": 0.03,

    # Gap: full için %2; analyze için %5
    "max_gap_full": 0.02,
    "max_gap_analyze": 0.05,

    # Outlier: full için %5; analyze için %10
    "max_outlier_full": 0.05,
    "max_outlier_analyze": 0.10,

    "allow_block_on_no_data": True,
}


# =========================
# Yardımcılar
# =========================

def _get(cfg: GuardrailsConfig, key: str) -> Any:
    return cfg[key] if key in cfg else DEFAULT_CFG[key]  # type: ignore

def _pct_or_zero(x: Any) -> float:
    try:
        if x is None: return 0.0
        return float(x)
    except Exception:
        return 0.0

def _years_or_zero(x: Any) -> float:
    try:
        if x is None: return 0.0
        return float(x)
    except Exception:
        return 0.0


# =========================
# Ana Karar Fonksiyonu
# =========================

def guard_or_route(
    quality: Dict[str, Any] | None,
    cfg: GuardrailsConfig | None,
    state: Dict[str, Any] | None = None,  # router log/ek bilgi için (opsiyonel)
) -> GuardDecision:
    """
    Veri kalitesini eşiklerle karşılaştırıp akış modunu belirler.

    Args:
        quality: data_quality.evaluate(df) çıktısı beklenir. Örn:
            {
              "coverage_years": 2.7,
              "nan_pct": 0.018,
              "duplicate_pct": 0.004,
              "gap_pct": 0.015,
              "outlier_ratio": 0.03,
              "coverage_class": "medium"
            }
        cfg: GuardrailsConfig (eşikleri aşmak için).
        state: Router state (opsiyonel, sadece log/kapsam için).

    Returns:
        GuardDecision:
            {
              "mode": "full" | "analyze" | "block",
              "reason": "coverage<1y" | "too_many_nans" | ...,
              "violations": ["...","..."]
            }
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}  # defaults + override
    violations: List[str] = []

    # 0) Force override (test/manuel)
    force_mode = cfg.get("force_mode")
    if force_mode in ("full", "analyze", "block"):
        return GuardDecision(mode=force_mode, reason="forced_by_cfg", violations=violations)

    # 1) Kalite raporu yoksa
    if not quality:
        if cfg.get("allow_block_on_no_data", True):
            return GuardDecision(mode="block", reason="no_quality_report", violations=["no_quality_report"])
        else:
            # Çok gevşek davranış: rapor yoksa analyze (risk kapalı) devam
            return GuardDecision(mode="analyze", reason="no_quality_report", violations=["no_quality_report"])

    # 2) Metrikleri çek
    years = _years_or_zero(quality.get("coverage_years"))
    nan_pct = _pct_or_zero(quality.get("nan_pct"))
    dup_pct = _pct_or_zero(quality.get("duplicate_pct"))
    gap_pct = _pct_or_zero(quality.get("gap_pct"))
    out_pct = _pct_or_zero(quality.get("outlier_ratio"))

    # 3) Öncelik: 'block' gerektiren ağır ihlaller
    #    (Her biri tek başına 'block' demek için yeterli)
    if years < _get(cfg, "min_years_analyze"):
        return GuardDecision(mode="block", reason="coverage<min_analyze", violations=["coverage<1y"])
    if nan_pct > _get(cfg, "max_nan_analyze"):
        return GuardDecision(mode="block", reason="too_many_nans", violations=[f"nan>{_get(cfg,'max_nan_analyze'):.2%}"])
    if dup_pct > _get(cfg, "max_duplicate_analyze"):
        return GuardDecision(mode="block", reason="too_many_duplicates", violations=[f"dup>{_get(cfg,'max_duplicate_analyze'):.2%}"])
    if gap_pct > _get(cfg, "max_gap_analyze"):
        return GuardDecision(mode="block", reason="too_many_gaps", violations=[f"gap>{_get(cfg,'max_gap_analyze'):.2%}"])
    if out_pct > _get(cfg, "max_outlier_analyze"):
        return GuardDecision(mode="block", reason="too_many_outliers", violations=[f"out>{_get(cfg,'max_outlier_analyze'):.2%}"])

    # 4) 'analyze' gerektiren orta dereceli ihlaller
    if years < _get(cfg, "min_years_full"):
        violations.append("coverage<min_full")
    if nan_pct > _get(cfg, "max_nan_full"):
        violations.append("nan>max_full")
    if dup_pct > _get(cfg, "max_duplicate_full"):
        violations.append("dup>max_full")
    if gap_pct > _get(cfg, "max_gap_full"):
        violations.append("gap>max_full")
    if out_pct > _get(cfg, "max_outlier_full"):
        violations.append("out>max_full")

    if violations:
        # Risk kapalı, analiz modunda devam (forecast/model/regime var; risk yok)
        return GuardDecision(mode="analyze", reason="degraded_quality", violations=violations)

    # 5) Hiç ihlal yok → full
    return GuardDecision(mode="full", reason=None, violations=violations)