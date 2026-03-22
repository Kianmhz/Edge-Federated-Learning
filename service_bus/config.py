"""
Centralized configuration for Azure Service Bus and Blob Storage.

All values are read from environment variables. When FL_TRANSPORT=http (the default),
the Service Bus layer is disabled and the system falls back to REST polling.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ServiceBusConfig:
    """Configuration for Azure Service Bus + Blob Storage integration."""

    # Transport mode: "servicebus" to enable, "http" to disable (default)
    transport: str = "http"

    # Azure Service Bus
    servicebus_connection_string: str = ""
    round_control_topic: str = "round-control"
    global_model_topic: str = "global-model"
    client_updates_queue: str = "client-updates"
    dashboard_events_queue: str = "dashboard-events"

    # Subscriptions
    round_control_subscription: str = "all-clients"
    global_model_subscription: str = "fl-clients"

    # Azure Blob Storage (for claim-check pattern)
    storage_connection_string: str = ""
    storage_container: str = "fl-model-weights"

    @property
    def enabled(self) -> bool:
        return self.transport == "servicebus"

    @classmethod
    def from_env(cls) -> "ServiceBusConfig":
        """Build config from environment variables."""
        return cls(
            transport=os.environ.get("FL_TRANSPORT", "http"),
            servicebus_connection_string=os.environ.get(
                "AZURE_SERVICEBUS_CONNECTION_STRING", ""
            ),
            round_control_topic=os.environ.get("SB_ROUND_CONTROL_TOPIC", "round-control"),
            global_model_topic=os.environ.get("SB_GLOBAL_MODEL_TOPIC", "global-model"),
            client_updates_queue=os.environ.get("SB_CLIENT_UPDATES_QUEUE", "client-updates"),
            dashboard_events_queue=os.environ.get("SB_DASHBOARD_EVENTS_QUEUE", "dashboard-events"),
            round_control_subscription=os.environ.get("SB_ROUND_CONTROL_SUB", "all-clients"),
            global_model_subscription=os.environ.get("SB_GLOBAL_MODEL_SUB", "fl-clients"),
            storage_connection_string=os.environ.get(
                "AZURE_STORAGE_CONNECTION_STRING", ""
            ),
            storage_container=os.environ.get("AZURE_STORAGE_CONTAINER", "fl-model-weights"),
        )

    def validate(self):
        """Raise if transport=servicebus but credentials are missing."""
        if not self.enabled:
            return
        if not self.servicebus_connection_string:
            raise ValueError(
                "FL_TRANSPORT=servicebus but AZURE_SERVICEBUS_CONNECTION_STRING is not set"
            )
        if not self.storage_connection_string:
            raise ValueError(
                "FL_TRANSPORT=servicebus but AZURE_STORAGE_CONNECTION_STRING is not set"
            )
