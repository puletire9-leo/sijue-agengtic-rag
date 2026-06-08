"""OpenTelemetry tracing setup for SuperMew."""
import atexit
import logging
import os
import signal
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

logger = logging.getLogger(__name__)


def setup_tracing():
    """Initialize OpenTelemetry tracing if OTEL_ENABLED=true."""
    if os.getenv("OTEL_ENABLED", "false").lower() != "true":
        return

    resource = Resource(attributes={SERVICE_NAME: "supermew"})
    provider = TracerProvider(resource=resource)

    endpoint = os.getenv("OTEL_EXPORTER_ENDPOINT", "http://localhost:4317")
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception:
        logger.warning("OTLP exporter not available, skipping span export")

    atexit.register(provider.shutdown)

    # SIGTERM handler — atexit won't fire on SIGTERM, so flush explicitly
    def _flush_on_sigterm(signum, frame):
        logger.info("SIGTERM received, flushing telemetry spans...")
        provider.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _flush_on_sigterm)

    trace.set_tracer_provider(provider)
