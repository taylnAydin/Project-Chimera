# router.py
"""
Chimera • Router (LangGraph Orchestrator)
- Tüm agent'leri sıralı/koşullu olarak çalıştırır.
- State tek gerçek kaynaktır; her node bir "reducer" gibi state'i günceller.
- Guardrails kararına göre akış 'full' / 'analyze' / 'block' dallarına ayrılır.
- Hataları toplayıp run'ı güvenle finalize eder.
"""

from __future__ import annotations
from typing import TypedDict, Literal, Any, Dict, List, Optional, Callable
from uuid import uuid4

# --- LangGraph ---
from langgraph.graph import StateGraph, START, END

# --- Agent modülleri ---
from . import data_retrieval as data_retrieval
from . import indicator as indicator
from . import data_quality as data_quality
from . import guardrails_agent as guardrails
from . import forecast as forecast
from . import model_selection as model_selection
from . import regime_detector as regime_detector
from . import risk as risk
from . import persistence as persistence
from . import reporter as reporter
from . import notifier as notifier
from . import backtester as backtester

# (Opsiyonel) anomaly mevcut değilse no-op
try:
    import agents.anomaly as anomaly
    _ANOMALY_AVAILABLE = True
except Exception:
    _ANOMALY_AVAILABLE = False
    anomaly = None  # type: ignore


# ======================================================================================
# Tipler & State
# ======================================================================================

class GuardDecision(TypedDict, total=False):
    mode: Literal["full", "analyze", "block"]
    reason: Optional[str]

class ChimeraState(TypedDict, total=False):
    # Kimlik & Girdi
    run_id: str
    symbol: str
    start: str
    end: str
    cfg: Dict[str, Any]

    # Artefaktlar
    df: Any
    quality: Dict[str, Any]
    forecasts: Dict[str, Any]
    best: Dict[str, Any]
    regime: Dict[str, Any]
    risk_plan: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    backtest: Dict[str, Any]
    eval_last: Dict[str, Any]            # <-- SON 10 GÜN eval çıktısı

    # Akış kontrol
    mode: Literal["full", "analyze", "block"]
    halt_reason: Optional[str]
    errors: List[str]
    failed: bool  # global fatal flag

    # Çıktı
    report: Dict[str, Any]


def make_initial_state(symbol: str, start: str, end: str, cfg: Dict[str, Any]) -> ChimeraState:
    """Router başlangıç state'ini oluşturur."""
    return ChimeraState(
        run_id="",
        symbol=symbol,
        start=start,
        end=end,
        cfg=cfg or {},
        df=None,
        quality={},
        forecasts={},
        best={},
        regime={},
        risk_plan={},
        anomalies=[],
        backtest={},
        eval_last={},              # <-- eklendi
        mode="full",
        halt_reason=None,
        errors=[],
        failed=False,
        report={}
    )


# ======================================================================================
# Yardımcılar
# ======================================================================================

SafeNode = Callable[[ChimeraState], ChimeraState]

def _append_error(state: ChimeraState, node_name: str, err: Exception) -> None:
    msg = f"{node_name}: {type(err).__name__}: {err}"
    state.setdefault("errors", []).append(msg)

def with_try(node_name: str) -> Callable[[SafeNode], SafeNode]:
    """
    Node'ları güvenli çalıştırmak için decorator.
    Hata olursa state.errors'a yazar ve state.failed=True yapar.
    """
    def deco(fn: SafeNode) -> SafeNode:
        def _wrapped(state: ChimeraState) -> ChimeraState:
            if state.get("failed"):
                return state  # zincir kırılmışsa hiçbir şey yapma
            try:
                return fn(state)
            except Exception as e:
                _append_error(state, node_name, e)
                state["failed"] = True
                return state
        return _wrapped  # type: ignore
    return deco


# ======================================================================================
# Node'lar (Reducer mantığı: State -> State)
# ======================================================================================

@with_try("start")
def node_start(state: ChimeraState) -> ChimeraState:
    state["run_id"] = str(uuid4())
    persistence.save_run_start(state["run_id"], state["symbol"], state["cfg"])
    return state

@with_try("data_retrieval")
def node_data_retrieval(state: ChimeraState) -> ChimeraState:
    df = data_retrieval.fetch(
        state["symbol"], state["start"], state["end"], state["cfg"].get("source", {})
    )
    state["df"] = df
    if df is None or len(df) == 0:
        state["mode"] = "block"
        state["halt_reason"] = "no_data"
    return state

@with_try("indicator")
def node_indicator(state: ChimeraState) -> ChimeraState:
    if state["mode"] == "block":
        return state
    state["df"] = indicator.enrich(state["df"], state["cfg"].get("indicators", {}))
    return state

@with_try("data_quality")
def node_data_quality(state: ChimeraState) -> ChimeraState:
    state["quality"] = data_quality.evaluate(state["df"])
    return state

@with_try("guardrails")
def node_guardrails(state: ChimeraState) -> ChimeraState:
    decision: GuardDecision = guardrails.guard_or_route(
        quality=state["quality"],
        cfg=state["cfg"].get("guardrails", {}),
        state=state
    )
    if "mode" in decision:
        state["mode"] = decision["mode"]  # type: ignore
    if "reason" in decision:
        state["halt_reason"] = decision["reason"]
    return state

@with_try("anomaly")
def node_anomaly(state: ChimeraState) -> ChimeraState:
    if state["mode"] == "block" or not _ANOMALY_AVAILABLE:
        return state
    state["anomalies"] = anomaly.detect(  # type: ignore
        state["df"], state.get("forecasts"), state["cfg"].get("anomaly", {})
    )
    return state

@with_try("forecast")
def node_forecast(state: ChimeraState) -> ChimeraState:
    if state["mode"] == "block":
        return state
    state["forecasts"] = forecast.make(state["df"], state["cfg"].get("forecast", {}))
    return state

@with_try("model_selection")
def node_model_selection(state: ChimeraState) -> ChimeraState:
    if state["mode"] == "block":
        return state
    state["best"] = model_selection.pick(
        state["forecasts"], metric=state["cfg"].get("model_select", {}).get("metric", "rmse")
    )
    return state

@with_try("eval_last")
def node_eval_last(state: ChimeraState) -> ChimeraState:
    """
    Seçilen modelin 'son 10 gün' dışörnek performansını üretir.
    forecast.evaluate_last_k_days(...) çağrısını yapar ve state['eval_last'] içine yazar.
    """
    if state["mode"] == "block":
        return state

    best = state.get("best", {}) or {}
    meta = best.get("meta", {}) or {}

    model_name = (meta.get("model") or best.get("name") or "ETS").lower()
    horizon = int(meta.get("horizon") or state["cfg"].get("forecast", {}).get("horizon", 7))

    exog_cols = None
    fc_cfg = state["cfg"].get("forecast", {}) or {}
    if isinstance(fc_cfg.get("exog"), dict):
        exog_cols = fc_cfg["exog"].get("cols")

    k = 10  # sabit: son 10 gün

    state["eval_last"] = forecast.evaluate_last_k_days(
        state["df"],
        model_name=model_name,
        horizon=horizon,
        exog_cols=exog_cols,
        k=k,
    )
    return state

@with_try("regime_detector")
def node_regime(state: ChimeraState) -> ChimeraState:
    if state["mode"] == "block":
        return state
    state["regime"] = regime_detector.detect(state["df"], state["cfg"].get("regime", {}))

    if state["cfg"].get("regime", {}).get("halt_on_extreme_vol") and \
       state["regime"].get("vol_class") == "extreme":
        state["mode"] = "analyze"
        state["halt_reason"] = "extreme_volatility"
    return state

@with_try("risk")
def node_risk(state: ChimeraState) -> ChimeraState:
    if state["mode"] != "full":
        return state
    state["risk_plan"] = risk.plan(state["best"], state["regime"], state["cfg"].get("risk", {}))
    return state

@with_try("backtester")
def node_backtest(state: ChimeraState) -> ChimeraState:
    """
    Amaç: Stratejiyi geçmiş veride test etmek.
    Koşul: mode!='block' (analyze veya full)
    Output: backtest (metrics, equity, trades)
    """
    if state["mode"] == "block":
        return state

    bt_cfg = state["cfg"].get("backtest", {})
    if state.get("risk_plan"):
        bt_cfg = {**bt_cfg, "risk": state["risk_plan"]}

    state["backtest"] = backtester.run(
        df=state["df"],
        forecasts=state.get("forecasts"),
        signals=None,
        cfg=bt_cfg
    )

    try:
        if hasattr(persistence, "save_backtest"):
            persistence.save_backtest(
                run_id=state["run_id"],
                symbol=state["symbol"],
                backtest=state["backtest"]
            )
    except Exception as e:
        _append_error(state, "backtester_persist", e)

    return state

@with_try("persistence_artifacts")
def node_persist(state: ChimeraState) -> ChimeraState:
    persistence.save_run_artifacts(
        run_id=state["run_id"],
        symbol=state["symbol"],
        df=state["df"],
        quality=state["quality"],
        forecasts=state["forecasts"],
        best=state["best"],
        regime=state["regime"],
        risk_plan=state["risk_plan"],
        anomalies=state["anomalies"],
        mode=state["mode"],
        halt_reason=state["halt_reason"],
    )
    return state

@with_try("reporter")
def node_reporter(state: ChimeraState) -> ChimeraState:
    state["report"] = reporter.build(
        run_id=state["run_id"],
        symbol=state["symbol"],
        best=state["best"],
        regime=state["regime"],
        risk_plan=state["risk_plan"],
        quality=state["quality"],
        anomalies=state["anomalies"],
        mode=state["mode"],
        backtest=state.get("backtest", {}),
        eval_last=state.get("eval_last", {}),   # <-- eval_last rapora iletiliyor
        cfg=state["cfg"].get("report", {})
    )
    return state

@with_try("notifier")
def node_notifier(state: ChimeraState) -> ChimeraState:
    notifier.send(state["report"], state["cfg"].get("notify", {}))
    return state

@with_try("end_success")
def node_end_success(state: ChimeraState) -> ChimeraState:
    persistence.save_run_end(state["run_id"], status="success")
    return state

@with_try("end_fail")
def node_end_fail(state: ChimeraState) -> ChimeraState:
    persistence.save_run_end(
        state.get("run_id", ""),
        status="failed",
        error=" | ".join(state.get("errors", [])) if state.get("errors") else None
    )
    return state


# ======================================================================================
# Koşullu geçiş fonksiyonları
# ======================================================================================

def branch_after_guardrails(state: ChimeraState) -> str:
    return "blocked" if state["mode"] == "block" else "ok"

def branch_after_regime(state: ChimeraState) -> str:
    if state["mode"] == "full":
        return "go_risk"
    elif state["mode"] == "analyze":
        return "go_backtest"
    else:
        return "skip_to_persist"

def branch_after_risk(state: ChimeraState) -> str:
    return "go_backtest"

def branch_end(state: ChimeraState) -> str:
    return "fail" if state.get("failed") else "ok"


# ======================================================================================
# Graph inşası & Çalıştırma
# ======================================================================================

def build_graph() -> Any:
    g = StateGraph(ChimeraState)

    # --- Node ekle
    g.add_node("start", node_start)
    g.add_node("data_retrieval", node_data_retrieval)
    g.add_node("indicator", node_indicator)
    g.add_node("data_quality", node_data_quality)
    g.add_node("guardrails", node_guardrails)
    if _ANOMALY_AVAILABLE:
        g.add_node("anomaly", node_anomaly)
    g.add_node("forecast", node_forecast)
    g.add_node("model_selection", node_model_selection)
    g.add_node("eval_last", node_eval_last)          # <-- eklendi
    g.add_node("regime", node_regime)
    g.add_node("risk", node_risk)
    g.add_node("backtest", node_backtest)
    g.add_node("persist", node_persist)
    g.add_node("reporter", node_reporter)
    g.add_node("notifier", node_notifier)
    g.add_node("end_ok", node_end_success)
    g.add_node("end_fail", node_end_fail)

    # --- Lineer akış başlangıcı
    g.add_edge(START, "start")
    g.add_edge("start", "data_retrieval")
    g.add_edge("data_retrieval", "indicator")
    g.add_edge("indicator", "data_quality")
    g.add_edge("data_quality", "guardrails")

    # --- Guardrails dalı
    g.add_conditional_edges(
        "guardrails",
        branch_after_guardrails,
        {
            "blocked": "persist",
            "ok": "anomaly" if _ANOMALY_AVAILABLE else "forecast"
        }
    )

    # --- Normal akış (anomali opsiyonel)
    if _ANOMALY_AVAILABLE:
        g.add_edge("anomaly", "forecast")
    g.add_edge("forecast", "model_selection")
    g.add_edge("model_selection", "eval_last")   # <-- eval_last araya
    g.add_edge("eval_last", "regime")
    g.add_edge("regime", "risk")

    # --- Rejim dalı
    g.add_conditional_edges(
        "regime",
        branch_after_regime,
        {
            "go_risk": "risk",
            "go_backtest": "backtest",
            "skip_to_persist": "persist"
        }
    )

    # risk → backtest
    g.add_conditional_edges(
        "risk",
        branch_after_risk,
        {
            "go_backtest": "backtest"
        }
    )

    # backtest → persist → reporter → notifier
    g.add_edge("backtest", "persist")
    g.add_edge("persist", "reporter")
    g.add_edge("reporter", "notifier")

    # --- Global finalize
    g.add_conditional_edges(
        "notifier",
        branch_end,
        {
            "ok": "end_ok",
            "fail": "end_fail"
        }
    )

    return g.compile()


def run_router(symbol: str, start: str, end: str, cfg: Dict[str, Any]) -> ChimeraState:
    """
    Router'ı tek çağrıyla çalıştırır ve final state'i döner.
    - Gemini çağrısı reporter.build() içinde yapılır.
    - Başarı/başarısız kapanış end_ok/end_fail nodelarında işlenir.
    """
    graph = build_graph()
    init = make_initial_state(symbol, start, end, cfg)
    final_state: ChimeraState = graph.invoke(init)
    return final_state