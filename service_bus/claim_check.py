"""
Claim-check pattern: store large payloads (model weights) in Azure Blob
Storage and pass a lightweight reference through Service Bus messages.

Blob naming convention:
  round-{round_id}/global-model          (server → clients)
  round-{round_id}/client-{client_id}    (client → server)
"""

from azure.storage.blob import BlobServiceClient, ContainerClient


class BlobClaimCheck:
    """Upload / download model weights to Azure Blob Storage."""

    def __init__(self, connection_string: str, container_name: str):
        self._blob_service = BlobServiceClient.from_connection_string(connection_string)
        self._container_name = container_name
        self._ensure_container()

    def _ensure_container(self):
        """Create the blob container if it doesn't exist."""
        container: ContainerClient = self._blob_service.get_container_client(
            self._container_name
        )
        if not container.exists():
            self._blob_service.create_container(self._container_name)

    # ---- public API ----

    def store(self, payload: bytes, blob_name: str) -> str:
        """Upload bytes and return the blob name (the 'claim ticket')."""
        blob_client = self._blob_service.get_blob_client(
            container=self._container_name, blob=blob_name
        )
        blob_client.upload_blob(payload, overwrite=True, max_concurrency=8)
        return blob_name

    def retrieve(self, blob_name: str) -> bytes:
        """Download and return blob content."""
        blob_client = self._blob_service.get_blob_client(
            container=self._container_name, blob=blob_name
        )
        return blob_client.download_blob().readall()

    def cleanup_old_blobs(self, current_round: int, keep_last: int = 3):
        """Delete blobs from rounds older than (current_round - keep_last).

        Non-fatal: errors are logged but do not propagate.
        """
        container = self._blob_service.get_container_client(self._container_name)
        cutoff = current_round - keep_last
        if cutoff < 0:
            return

        try:
            for blob in container.list_blobs():
                # blob names start with "round-{N}/"
                parts = blob.name.split("/")
                if len(parts) >= 2 and parts[0].startswith("round-"):
                    try:
                        round_num = int(parts[0].split("-", 1)[1])
                        if round_num < cutoff:
                            container.delete_blob(blob.name)
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            print(f"[BLOB] Cleanup warning: {e}")

    def close(self):
        self._blob_service.close()
