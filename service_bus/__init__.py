"""
Azure Service Bus integration for Federated Learning pipeline.

Replaces HTTP polling with event-driven pub/sub messaging:
  - Topic 'round-control': server broadcasts round start + selected clients
  - Topic 'global-model': server publishes aggregated model via claim-check pattern
  - Queue 'client-updates': clients send weight deltas to server
  - Queue 'dashboard-events': server pushes live events for dashboard
"""

from service_bus.config import ServiceBusConfig
from service_bus.manager import ServiceBusManager
from service_bus.claim_check import BlobClaimCheck
from service_bus.serialization import serialize_weights, deserialize_weights

__all__ = [
    "ServiceBusConfig",
    "ServiceBusManager",
    "BlobClaimCheck",
    "serialize_weights",
    "deserialize_weights",
]
