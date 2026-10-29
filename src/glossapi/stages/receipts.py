"""Request-scoped transport evidence, independent of SQLite and legacy stores."""
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import time
import traceback

_transport_receipts = ContextVar("glossapi_transport_receipts", default=None)
_current_request = ContextVar("glossapi_http_request", default=None)


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolved_route(route, options):
    state = _transport_receipts.get()
    if state is not None:
        from ..ledger.encoding import canonical_json
        state["route"] = route
        state["policy_hash"] = hashlib.sha256(canonical_json({"configuration_hash": state["policy_hash"], "resolved_options": options}).encode()).hexdigest()


@contextmanager
def capture_transport(route, policy):
    from ..ledger.encoding import canonical_json
    state = {"route": route, "policy_hash": hashlib.sha256(canonical_json(policy).encode()).hexdigest(),
             "requests": [], "acquisitions": [], "source_cap": policy.get("max_source_bytes")}
    token = _transport_receipts.set(state)
    try:
        yield state
    finally:
        _transport_receipts.reset(token)


@contextmanager
def request_receipt(url, *, retry_index=0, backoff_ms=None, route=None):
    state = _transport_receipts.get()
    if state is None:
        yield None
        return
    receipt = {"sequence": len(state["requests"]), "requested_url": url, "final_url": None,
        "http_status": None, "redirects": [], "response_headers": {}, "retry_index": retry_index,
        "route": route or state["route"], "policy_hash": state["policy_hash"], "started_at": utc_now(),
        "finished_at": None, "backoff_ms": backoff_ms, "retry_after_ms": None, "error": None,
        "source_entity_bytes": None, "wire_bytes": None}
    state["requests"].append(receipt)
    token = _current_request.set(receipt)
    try:
        yield receipt
    except BaseException as error:
        receipt["error"] = {"error_type": type(error).__module__ + "." + type(error).__name__,
            "message": str(error), "traceback": "".join(traceback.format_exception(error))}
        raise
    finally:
        receipt["finished_at"] = utc_now()
        _current_request.reset(token)


def response_receipt(response, requested_url, retry_index=None):
    state = _transport_receipts.get()
    if state is None:
        return
    receipt = _current_request.get()
    if receipt is None:
        # Browser/bootstrap requests that expose only the response have truthful
        # nullable starts; do not invent a duration or request start time.
        with request_receipt(requested_url, retry_index=retry_index, route=state["route"]) as receipt:
            receipt["started_at"] = None
            _response_fields(receipt, response)
    else:
        _response_fields(receipt, response)


def _response_fields(receipt, response):
    headers = {str(key).lower(): str(value) for key, value in (getattr(response, "headers", {}) or {}).items()
        if str(key).lower() in ("content-type", "content-disposition", "content-length", "content-encoding", "etag", "last-modified", "retry-after")}
    receipt.update(final_url=str(getattr(response, "url", receipt["requested_url"])), http_status=getattr(response, "status", None),
        response_headers=headers, redirects=[{"url": str(item.url), "status": item.status,
            "location": item.headers.get("Location")} for item in getattr(response, "history", ())])
    value = headers.get("retry-after", "")
    if value.isdigit():
        receipt["retry_after_ms"] = int(value) * 1000
    elif value:
        from email.utils import parsedate_to_datetime
        try:
            receipt["retry_after_ms"] = max(0, int((parsedate_to_datetime(value).timestamp()-time.time())*1000))
        except (ValueError, TypeError, OverflowError):
            pass


def acquired_receipt(url, path, source_observable):
    state = _transport_receipts.get()
    if state is None:
        return
    from .execution import sha256_file
    digest, size = sha256_file(path), path.stat().st_size
    state["acquisitions"].append({"requested_url": url, "stored_sha256": digest, "stored_bytes": size,
        "source_sha256": digest if source_observable else None, "source_entity_bytes": size if source_observable else None,
        "wire_bytes": None, "source_hash_unavailable_reason": None if source_observable else "original_entity_not_observable",
        "transformation": "identity" if source_observable else "browser_or_source_specific_recovery"})


@asynccontextmanager
async def recorded_request(session, method, url, *, retry_index=0, route=None, **kwargs):
    # Match tenacity's configured wait_exponential(1, min=1, max=10).
    backoff = 0 if retry_index == 0 else min(10, 2 ** (retry_index-1)) * 1000
    with request_receipt(url, retry_index=retry_index, backoff_ms=backoff, route=route):
        async with getattr(session, method)(url, **kwargs) as response:
            yield response


def received_bytes(count):
    receipt = _current_request.get()
    if receipt is not None:
        amount=(receipt['source_entity_bytes'] or 0)+count
        state=_transport_receipts.get()
        cap=state.get('source_cap') if state else None
        if cap is not None and amount>cap:
            from .disk_budget import DiskBudgetError
            raise DiskBudgetError('disk_budget_exceeded','download exceeds max_source_bytes before writing the next chunk')
        receipt['source_entity_bytes']=amount


@contextmanager
def capture_backend_observations():
    from ..ledger.journal import WorkerObservations, _active_journal
    collector = WorkerObservations()
    token = _active_journal.set(collector)
    try:
        yield collector.receipts
    finally:
        _active_journal.reset(token)
