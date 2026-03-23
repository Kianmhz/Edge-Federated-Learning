"""
ServiceBusManager — central facade for all Azure Service Bus operations.

Server-side:
  - publish_round_control()   → Topic 'round-control'
  - publish_global_model()    → Topic 'global-model' (claim-check)
  - publish_dashboard_event() → Queue 'dashboard-events'
  - receive_client_updates()  → Queue 'client-updates' (blocking loop)

Client-side:
  - wait_for_round_control()  → Subscribe to 'round-control'
  - receive_global_model()    → Subscribe to 'global-model' (claim-check)
  - submit_client_update()    → Send to 'client-updates' queue
"""

import json
import time
from datetime import datetime

from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.servicebus.management import ServiceBusAdministrationClient

from service_bus.config import ServiceBusConfig
from service_bus.claim_check import BlobClaimCheck
from service_bus.serialization import serialize_weights, deserialize_weights


class ServiceBusManager:
    """Unified interface for Service Bus messaging + Blob claim-check."""

    def __init__(self, config: ServiceBusConfig):
        self._config = config
        self._sb_client = ServiceBusClient.from_connection_string(
            config.servicebus_connection_string
        )
        self._admin_client = ServiceBusAdministrationClient.from_connection_string(
            config.servicebus_connection_string
        )
        self._blob = BlobClaimCheck(
            config.storage_connection_string, config.storage_container
        )
        self._ensured_subscriptions: set = set()

    def _ensure_subscription(self, topic_name: str, subscription_name: str):
        """Create a topic subscription if it doesn't already exist."""
        key = (topic_name, subscription_name)
        if key in self._ensured_subscriptions:
            return
        try:
            self._admin_client.get_subscription(topic_name, subscription_name)
        except Exception:
            try:
                self._admin_client.create_subscription(topic_name, subscription_name)
                print(f"[SB] Created subscription '{subscription_name}' on topic '{topic_name}'")
            except Exception as e:
                # May already exist due to a race — that's fine
                print(f"[SB] Subscription '{subscription_name}' on '{topic_name}': {e}")
        self._ensured_subscriptions.add(key)

    # ------------------------------------------------------------------
    # SERVER-SIDE: Publishing
    # ------------------------------------------------------------------

    def publish_round_control(self, round_id: int, selected_clients: list):
        """Broadcast a round-start event with the list of selected clients.

        Published to the 'round-control' topic. All client subscriptions
        receive the message; each client filters by its own ID in the body.
        """
        body = json.dumps({
            "round": round_id,
            "selected_clients": selected_clients,
            "timestamp": datetime.now().isoformat(),
        })
        msg = ServiceBusMessage(
            body=body,
            application_properties={
                "round_id": round_id,
                "event_type": "round_start",
            },
        )

        sender = self._sb_client.get_topic_sender(
            topic_name=self._config.round_control_topic
        )
        with sender:
            sender.send_messages(msg)

        print(f"[SB] Published round-control for round {round_id}: {selected_clients}")

    def publish_global_model(self, round_id: int, state_dict: dict):
        """Publish the aggregated global model via claim-check pattern.

        1. Serialize weights → bytes
        2. Upload to Blob Storage
        3. Send a lightweight reference message on the 'global-model' topic
        """
        blob_name = f"round-{round_id}/global-model"
        payload = serialize_weights(state_dict)
        self._blob.store(payload, blob_name)

        body = json.dumps({
            "round": round_id,
            "blob_name": blob_name,
            "timestamp": datetime.now().isoformat(),
        })
        msg = ServiceBusMessage(
            body=body,
            application_properties={
                "round_id": round_id,
                "event_type": "global_model",
            },
        )

        sender = self._sb_client.get_topic_sender(
            topic_name=self._config.global_model_topic
        )
        with sender:
            sender.send_messages(msg)

        # Cleanup old blobs
        self._blob.cleanup_old_blobs(round_id)
        print(f"[SB] Published global-model for round {round_id} (blob: {blob_name})")

    def publish_dashboard_event(self, event: dict):
        """Push a lightweight JSON event to the dashboard-events queue."""
        body = json.dumps(event)
        msg = ServiceBusMessage(body=body)

        sender = self._sb_client.get_queue_sender(
            queue_name=self._config.dashboard_events_queue
        )
        with sender:
            sender.send_messages(msg)

    # ------------------------------------------------------------------
    # SERVER-SIDE: Receiving client updates
    # ------------------------------------------------------------------

    def receive_client_updates(self, callback, poll_interval: float = 1.0):
        """Blocking loop: receive client weight updates from the queue.

        Each message is expected to contain:
          {"client_id": str, "round": int, "blob_name": str, "data_size": int}

        The callback receives (client_id, round_id, weights_dict, data_size).
        Designed to run in a background thread.
        """
        receiver = self._sb_client.get_queue_receiver(
            queue_name=self._config.client_updates_queue,
            max_wait_time=5,
        )
        with receiver:
            while True:
                try:
                    messages = receiver.receive_messages(
                        max_message_count=10, max_wait_time=5
                    )
                    for msg in messages:
                        try:
                            data = json.loads(str(msg))
                            client_id = data["client_id"]
                            round_id = data["round"]
                            blob_name = data["blob_name"]
                            data_size = data["data_size"]

                            # Retrieve weights from blob
                            weight_bytes = self._blob.retrieve(blob_name)
                            weights = deserialize_weights(weight_bytes)

                            callback(client_id, round_id, weights, data_size)
                            receiver.complete_message(msg)
                        except Exception as e:
                            print(f"[SB] Error processing update message: {e}")
                            try:
                                receiver.dead_letter_message(
                                    msg, reason="ProcessingError", error_description=str(e)
                                )
                            except Exception:
                                receiver.complete_message(msg)

                    if not messages:
                        time.sleep(poll_interval)
                except Exception as e:
                    print(f"[SB] Receiver error: {e}")
                    time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # CLIENT-SIDE: Subscribing
    # ------------------------------------------------------------------

    def ensure_client_subscriptions(self, client_id: str):
        """Pre-create per-client subscriptions on both topics.

        Must be called before entering the client loop so that subscriptions
        exist when the server publishes messages. Messages published to a topic
        before a subscription exists are never retroactively delivered.
        """
        self._ensure_subscription(self._config.round_control_topic, f"client-{client_id}")
        self._ensure_subscription(self._config.global_model_topic, f"client-{client_id}")

    def wait_for_round_control(self, client_id: str, timeout: float = 60.0):
        """Block until a round-control message selects this client.

        Returns (round_id, selected_clients) or (None, None) on timeout.
        Each client uses its own subscription so messages are not stolen by
        competing consumers.
        """
        subscription_name = f"client-{client_id}"
        self._ensure_subscription(self._config.round_control_topic, subscription_name)

        receiver = self._sb_client.get_subscription_receiver(
            topic_name=self._config.round_control_topic,
            subscription_name=subscription_name,
            max_wait_time=int(timeout),
        )
        with receiver:
            start = time.time()
            while time.time() - start < timeout:
                remaining = max(1, int(timeout - (time.time() - start)))
                messages = receiver.receive_messages(
                    max_message_count=1, max_wait_time=remaining
                )
                for msg in messages:
                    try:
                        data = json.loads(str(msg))
                        round_id = data["round"]
                        selected = data["selected_clients"]
                        receiver.complete_message(msg)

                        if str(client_id) in [str(c) for c in selected]:
                            return round_id, selected
                        # Not selected — continue waiting
                    except Exception as e:
                        print(f"[SB] Error parsing round-control: {e}")
                        receiver.complete_message(msg)

        return None, None

    def receive_global_model(self, client_id: str, timeout: float = 30.0):
        """Block until the global model is available on the topic.

        Returns (round_id, weights_dict) or (None, None) on timeout.
        Uses claim-check: downloads weights from Blob Storage.
        Each client uses its own subscription to receive its own copy.
        """
        subscription_name = f"client-{client_id}"
        self._ensure_subscription(self._config.global_model_topic, subscription_name)

        receiver = self._sb_client.get_subscription_receiver(
            topic_name=self._config.global_model_topic,
            subscription_name=subscription_name,
            max_wait_time=int(timeout),
        )
        with receiver:
            messages = receiver.receive_messages(
                max_message_count=1, max_wait_time=int(timeout)
            )
            for msg in messages:
                try:
                    data = json.loads(str(msg))
                    round_id = data["round"]
                    blob_name = data["blob_name"]

                    weight_bytes = self._blob.retrieve(blob_name)
                    weights = deserialize_weights(weight_bytes)
                    receiver.complete_message(msg)
                    return round_id, weights
                except Exception as e:
                    print(f"[SB] Error receiving global model: {e}")
                    receiver.complete_message(msg)

        return None, None

    def submit_client_update(
        self, client_id: str, round_id: int, weights: dict, data_size: int
    ):
        """Send a weight update to the server via the client-updates queue.

        Uses claim-check: uploads weights to Blob, sends reference message.
        """
        blob_name = f"round-{round_id}/client-{client_id}"
        payload = serialize_weights(weights)
        self._blob.store(payload, blob_name)

        body = json.dumps({
            "client_id": str(client_id),
            "round": round_id,
            "blob_name": blob_name,
            "data_size": data_size,
        })
        msg = ServiceBusMessage(
            body=body,
            application_properties={
                "client_id": str(client_id),
                "round_id": round_id,
            },
        )

        sender = self._sb_client.get_queue_sender(
            queue_name=self._config.client_updates_queue
        )
        with sender:
            sender.send_messages(msg)

        print(f"[SB] Client {client_id} submitted update for round {round_id}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Release Service Bus and Blob Storage connections."""
        try:
            self._sb_client.close()
        except Exception:
            pass
        try:
            self._admin_client.close()
        except Exception:
            pass
        try:
            self._blob.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
