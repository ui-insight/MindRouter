############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# otel.py: OpenTelemetry initialization and configuration
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""OpenTelemetry tracing and metrics setup.

Gated behind ``OTEL_ENABLED=true``.  When disabled, all OTel calls
become no-ops (the API provides a no-op tracer/meter by default).
"""

import logging
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

logger = logging.getLogger(__name__)

_tracer_provider: Optional[TracerProvider] = None
_meter_provider: Optional[MeterProvider] = None


def setup_telemetry() -> None:
    """Initialize OpenTelemetry tracing and metrics.

    Reads configuration from ``settings.py``.  Safe to call even when
    ``otel_enabled`` is False — in that case it does nothing and the
    default no-op providers remain active.
    """
    global _tracer_provider, _meter_provider

    from backend.app.settings import get_settings
    settings = get_settings()

    if not settings.otel_enabled:
        logger.info("OpenTelemetry disabled (OTEL_ENABLED=false)")
        return

    endpoint = settings.otel_exporter_otlp_endpoint
    if not endpoint:
        logger.warning(
            "OTEL_ENABLED=true but OTEL_EXPORTER_OTLP_ENDPOINT is not set — "
            "skipping OTel initialization"
        )
        return

    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": settings.app_version,
    })

    # --- Tracing ---
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(_tracer_provider)

    # --- Metrics ---
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

    metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30000)
    _meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(_meter_provider)

    # --- Auto-instrumentation ---
    _instrument_libraries()

    logger.info(
        "OpenTelemetry initialized",
        extra={"endpoint": endpoint, "service": settings.otel_service_name},
    )


def _instrument_libraries() -> None:
    """Apply auto-instrumentation to FastAPI, httpx, SQLAlchemy, and Redis."""
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    # httpx — instruments all outbound HTTP calls to backends
    HTTPXClientInstrumentor().instrument()

    # Redis
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        RedisInstrumentor().instrument()
    except Exception:
        logger.debug("Redis instrumentation skipped (redis may not be configured)")

    # SQLAlchemy — instrument the engine after it's created.
    # The engine is created lazily in db/session.py, so we instrument
    # the module-level engine reference.
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from backend.app.db.session import engine
        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
    except Exception:
        logger.debug("SQLAlchemy instrumentation skipped", exc_info=True)


def instrument_app(app) -> None:
    """Apply FastAPI auto-instrumentation to the app instance.

    Called after the app is created so the instrumentor can hook into
    the ASGI middleware stack.
    """
    from backend.app.settings import get_settings
    if not get_settings().otel_enabled:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry")
    except Exception:
        logger.warning("FastAPI OTel instrumentation failed", exc_info=True)


def shutdown_telemetry() -> None:
    """Flush and shut down OTel providers.  Safe to call even if not initialized."""
    global _tracer_provider, _meter_provider

    if _tracer_provider:
        _tracer_provider.shutdown()
        _tracer_provider = None

    if _meter_provider:
        _meter_provider.shutdown()
        _meter_provider = None

    logger.info("OpenTelemetry shut down")


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance.  Returns a no-op tracer when OTel is disabled."""
    return trace.get_tracer(name)
