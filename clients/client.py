import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import requests
import torch

from clients.data_utils import get_client_loaders, get_dataloader
from clients.training import train_local
from fl_core.model_def import get_model

SERVER_URL = "http://127.0.0.1:9000"


def _serialize_state(state_dict):
    return {k: v.cpu().tolist() for k, v in state_dict.items()}


# Compute the update (local - global), clip its global L2 norm, and add Gaussian noise if DP enabled.
def _add_dp_to_update(
    global_state, local_state, clip_norm=1.0, noise_multiplier=0.0, dp_enabled=False
):
    update = {k: (local_state[k] - global_state[k]).float() for k in local_state.keys()}

    if not dp_enabled:
        return _serialize_state({k: v for k, v in update.items()})

    # Flatten and compute global L2 norm
    squared = 0.0
    for v in update.values():
        squared += float((v.double() ** 2).sum().item())
    global_norm = squared**0.5

    # clip if necessary
    if global_norm > clip_norm and global_norm > 0:
        clip_coef = clip_norm / global_norm
    else:
        clip_coef = 1.0

    clipped_update = {k: (v * clip_coef) for k, v in update.items()}

    # add Gaussian noise to each parameter tensor
    stddev = noise_multiplier * clip_norm
    noised_update = {}
    for k, v in clipped_update.items():
        if stddev > 0:
            noise = torch.normal(
                mean=0.0, std=stddev, size=v.size(), device=v.device, dtype=v.dtype
            )
            noised = v + noise
        else:
            noised = v
        noised_update[k] = noised

    return _serialize_state(noised_update)


def run_client_loop_sb(
    train_loader,
    client_id,
    sb_manager,
    dp_enabled=False,
    noise_multiplier=0.0,
    clip_norm=1.0,
    debug=False,
):
    """Event-driven client loop using Azure Service Bus.

    Instead of polling HTTP endpoints, the client:
      1. Waits for a round-control message (topic subscription)
      2. Receives the global model via claim-check (topic + blob)
      3. Trains locally
      4. Submits the weight delta via claim-check (queue + blob)
      5. Waits for the next round-control message (no polling needed)
    """
    print(f"[Client {client_id}] Running in Service Bus transport mode")

    while True:
        # --- Phase 1: Register with server to trigger round selection ---
        # The server only publishes round-control when it knows clients are connected.
        # Calling /should_participate registers this client and triggers the SB publish.
        try:
            requests.get(f"{SERVER_URL}/should_participate", params={"client_id": client_id}, timeout=5)
        except Exception:
            pass  # Non-fatal: server may not be reachable, SB message may already be published

        # --- Phase 1b: Wait for round selection via Service Bus ---
        print(f"[Client {client_id}] Waiting for round-control message...")
        round_id, selected = sb_manager.wait_for_round_control(
            client_id, timeout=300.0
        )

        if round_id is None:
            print(f"[Client {client_id}] Timed out waiting for round-control. Retrying...")
            continue

        print(f"[Client {client_id}] Selected for round {round_id}. Receiving global model...")

        # --- Phase 2: Receive global model via claim-check ---
        model_round, weights = sb_manager.receive_global_model(timeout=30.0)

        if weights is None:
            # Fallback: fetch via HTTP if Service Bus delivery fails
            print(f"[Client {client_id}] Service Bus model delivery failed, falling back to HTTP...")
            try:
                r = requests.get(f"{SERVER_URL}/get_model").json()
                weights = {k: torch.tensor(v) for k, v in r["weights"].items()}
            except Exception as e:
                print(f"[Client {client_id}] HTTP fallback also failed: {e}. Skipping round.")
                continue

        # Load weights into model
        model = get_model()
        # weights from blob are already tensors; from HTTP fallback they're also tensors
        model.load_state_dict(weights)

        # --- Phase 3: Train locally ---
        print(f"[Client {client_id}] Training locally...")
        global_state = {k: v.clone().float() for k, v in weights.items()}
        model = train_local(model, train_loader, epochs=1)
        local_state = {k: v.clone().float() for k, v in model.state_dict().items()}

        # --- Phase 4: Compute update delta (with optional DP) ---
        if dp_enabled and debug:
            raw_update = {
                k: (local_state[k] - global_state[k]).float()
                for k in global_state.keys()
            }

        weights_to_send = _add_dp_to_update(
            global_state, local_state,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            dp_enabled=dp_enabled,
        )

        # --- Phase 5: Submit via Service Bus claim-check ---
        print(f"[Client {client_id}] Sending update via Service Bus... (DP={'ON' if dp_enabled else 'OFF'})")
        try:
            sb_manager.submit_client_update(
                client_id=str(client_id),
                round_id=round_id,
                weights=weights_to_send,
                data_size=len(train_loader.dataset),
            )
            print(f"[Client {client_id}] Update sent for round {round_id}. Waiting for next round...\n")
        except Exception as e:
            print(f"[Client {client_id}] Service Bus submit failed: {e}")

        # No polling needed — the next iteration blocks on wait_for_round_control


def run_client_loop(
    train_loader,
    client_id,
    dp_enabled=False,
    noise_multiplier=0.0,
    clip_norm=1.0,
    debug=False,
):
    """Run the main client loop using the provided train_loader."""
    while True:
        # First: ask server whether this client should participate this round
        try:
            sel = requests.get(f"{SERVER_URL}/should_participate", params={"client_id": client_id}).json()
            selected = bool(sel.get("selected", False))
            server_round = sel.get("round", None)
        except Exception as e:
            print(f"[Client {client_id}] Failed to fetch participation decision: {e}. Defaulting to participate.")
            selected = True
            server_round = None

        if not selected:
            print(f"[Client {client_id}] Not selected for round {server_round}. Sleeping until next check...")

            time.sleep(3)
            continue

        print(f"[Client {client_id}] Selected for round {server_round}. Fetching global model...")
        r = requests.get(f"{SERVER_URL}/get_model").json()

        # Load server weights
        model = get_model()
        state = {k: torch.tensor(v) for k, v in r["weights"].items()}
        model.load_state_dict(state)

        print(f"[Client {client_id}] Training locally...")
        model = train_local(model, train_loader, epochs=1)

        # Prepare update: either raw weights or DP-processed (delta, clip, noise)
        # We compute update = local - global and send that (so server can aggregate deltas)
        global_state = {k: v.clone().float() for k, v in state.items()}
        local_state = {k: v.clone().float() for k, v in model.state_dict().items()}

        # If debug and dp enabled, compute raw stats
        if dp_enabled and debug:
            raw_update = {
                k: (local_state[k] - global_state[k]).float()
                for k in global_state.keys()
            }

        weights_to_send = _add_dp_to_update(
            global_state,
            local_state,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            dp_enabled=dp_enabled,
        )

        # If debug and dp enabled, compute post stats and print both
        if dp_enabled and debug:
            try:
                # compute global_norm and clip coef similar to _add_dp_to_update
                squared = 0.0
                for v in raw_update.values():
                    squared += float((v.double() ** 2).sum().item())
                global_norm = squared**0.5
                clip_coef = (
                    (clip_norm / global_norm)
                    if (global_norm > 0 and global_norm > clip_norm)
                    else 1.0
                )
                stddev = noise_multiplier * clip_norm

                # per-tensor pre/post stats for up to 3 keys
                keys = list(raw_update.keys())[:3]
                per_tensor_stats = []
                for k in keys:
                    pre_t = raw_update[k].view(-1)
                    pre_sample = pre_t[:5].cpu().tolist()
                    pre_norm = float(pre_t.norm().item())
                    post_t = torch.tensor(weights_to_send[k]).view(-1)
                    post_sample = post_t[:5].cpu().tolist()
                    post_norm = float(post_t.norm().item())
                    per_tensor_stats.append(
                        (k, pre_sample, pre_norm, post_sample, post_norm)
                    )

                # Compact table-style debug output
                header = f"{'key':25} {'#params':>8} {'PRE_norm':>10} {'EXP_noise':>12} {'POST_norm':>12}"
                print("\n--- CLIENT DP DEBUG ---")
                print(
                    f"global_flat_norm: {global_norm:.6f} | clip_norm: {clip_norm} | clip_coef: {clip_coef:.6f} | noise_std: {stddev}"
                )
                print(header)
                print("-" * len(header))
                total_params = 0
                total_expected_noise = 0.0
                for k, _, pre_n, _, post_n in per_tensor_stats:
                    numel = int(raw_update[k].numel())
                    expected_noise = stddev * (numel**0.5)
                    total_params += numel
                    total_expected_noise += expected_noise
                    print(
                        f"{k:25} {numel:8d} {pre_n:10.4f} {expected_noise:12.4f} {post_n:12.4f}"
                    )

                print(
                    f"Total params shown: {total_params} | Total expected_noise (approx): {total_expected_noise:.4f}"
                )
                print("--- END DEBUG ---\n")
            except Exception as e:
                print(f"[Client DEBUG] failed to compute detailed stats: {e}")

        print(
            f"[Client {client_id}] Sending update... (DP={'ON' if dp_enabled else 'OFF'})"
        )
        # Include the round id the client believes it's participating in so
        # the server can reject stale/duplicate submissions.
        submit_payload = {
            "client_id": str(client_id),
            "weights": weights_to_send,
            "data_size": len(train_loader.dataset),
            "round": server_round,
        }

        try:
            resp = requests.post(f"{SERVER_URL}/submit_update", json=submit_payload)
            resp_json = resp.json() if resp is not None else {}
        except Exception as e:
            print(f"[Client {client_id}] Failed to submit update: {e}")
            resp_json = {}

        # After submitting, wait until server advances to a newer round than
        # the one we just submitted for. This prevents a fast client from
        # submitting multiple updates for the same server round.
        print(f"[Client {client_id}] Update sent. Waiting for server to advance from round {server_round}...\n")

        # Poll server's /get_model endpoint for the current round. Timeout
        # after a reasonable period so client can retry participation checks.
        wait_start = time.time()
        timeout = 60  # seconds
        while True:
            try:
                status = requests.get(f"{SERVER_URL}/config").json()
                server_now = status.get("current_round", None)
            except Exception:
                server_now = None

            if server_now is not None and server_now > server_round:
                # server completed aggregation and moved to next round
                break

            if time.time() - wait_start > timeout:
                # give up waiting after timeout; loop back and check selection
                print(f"[Client {client_id}] Waited {timeout}s for server to advance; re-checking participation...")
                break

            time.sleep(1)


def client_loop(
    client_id, dp_enabled=False, noise_multiplier=0.0, clip_norm=1.0, debug=False
):
    # default: load full dataset (backwards-compatible). If caller set CLIENT_LOADERS
    # externally they can pass a per-client loader via closure; here we default to full loader.
    train_loader, _ = get_dataloader()
    run_client_loop(
        train_loader,
        client_id,
        dp_enabled=dp_enabled,
        noise_multiplier=noise_multiplier,
        clip_norm=clip_norm,
        debug=debug,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated client with optional DP on updates"
    )
    parser.add_argument("client_id", nargs="?", default="1", help="Client id string")
    parser.add_argument(
        "--dp", action="store_true", help="Enable differential privacy (clip + noise)"
    )
    parser.add_argument(
        "--noise", type=float, default=0.05, help="Noise multiplier (std = noise * clip)"
    )
    parser.add_argument("--clip", type=float, default=1.0, help="Clip norm for DP (L2)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable client-side debug prints (works with --dp)",
    )
    # options to control client-side data partitioning
    # num_clients removed from CLI to avoid mismatch with server config;
    # client will fetch num_clients from server /config at startup.
    parser.add_argument(
        "--non-iid",
        action="store_true",
        help="Use Dirichlet non-IID split instead of IID",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha parameter (smaller -> more skew)",
    )
    parser.add_argument(
        "--transport",
        choices=["http", "servicebus"],
        default=os.environ.get("FL_TRANSPORT", "http"),
        help="Transport mode: 'http' (REST polling) or 'servicebus' (Azure Service Bus)",
    )

    args = parser.parse_args()

    # Always fetch num_clients from server config before printing split info
    try:
        cfg = requests.get(f"{SERVER_URL}/config").json()
        num_clients = int(cfg.get("num_clients", 0)) or None
        print(f"[CLIENT] Fetched num_clients={num_clients} from server config")
    except Exception as e:
        num_clients = None
        print(f"[CLIENT] Failed to fetch server config: {e}. Falling back to default loader behavior")

    # Initialize Service Bus manager if transport=servicebus
    sb_mgr = None
    if args.transport == "servicebus":
        from service_bus.config import ServiceBusConfig
        from service_bus.manager import ServiceBusManager
        sb_cfg = ServiceBusConfig.from_env()
        sb_cfg.validate()
        sb_mgr = ServiceBusManager(sb_cfg)
        print(f"[CLIENT] Azure Service Bus transport ENABLED")

    num_clients_display = num_clients if num_clients is not None else "unknown"
    # print whether we're running IID or non-IID
    if args.non_iid:
        print(
            f"[CLIENT] Running with non-IID Dirichlet split (num_clients={num_clients_display}, alpha={args.alpha})"
        )
    else:
        print(f"[CLIENT] Running with IID split (num_clients={num_clients_display})")

    if num_clients is not None:
        clients_loaders, _ = get_client_loaders(
            num_clients=num_clients, iid=(not args.non_iid), alpha=args.alpha
        )

        # replace get_dataloader behavior by injecting local loader into client_loop via closure
        def client_loop_with_local_loader(
            client_id,
            dp_enabled=False,
            noise_multiplier=0.0,
            clip_norm=1.0,
            debug=False,
        ):
            try:
                cid = int(client_id)
            except Exception:
                cid = 0
            # Map 1-indexed client IDs to 0-indexed partitions
            partition_idx = (cid - 1) % num_clients if num_clients else cid
            train_loader = clients_loaders.get(partition_idx, None)
            # Validate requested client id if server provided it
            if num_clients is not None and (cid < 1):
                print(
                    f"[CLIENT] ERROR: client_id={cid} invalid. Falling back to full loader."
                )
                train_loader, _ = get_dataloader()
            else:
                # report selected partition size
                if train_loader is not None:
                    try:
                        print(
                            f"[CLIENT] Loaded partition {partition_idx} for client {cid}: {len(train_loader.dataset)} samples"
                        )
                    except Exception:
                        print(f"[CLIENT] Loaded partition {partition_idx} for client {cid}")
            if train_loader is None:
                # fallback to full loader
                print(
                    f"[Client {client_id}] WARNING: no local loader found, using full dataset loader"
                )
                train_loader, _ = get_dataloader()

            # Dispatch to Service Bus or HTTP loop
            loop_fn = run_client_loop_sb if sb_mgr else run_client_loop
            loop_kwargs = dict(
                train_loader=train_loader,
                client_id=client_id,
                dp_enabled=dp_enabled,
                noise_multiplier=noise_multiplier,
                clip_norm=clip_norm,
                debug=debug,
            )
            if sb_mgr:
                loop_kwargs["sb_manager"] = sb_mgr
            loop_fn(**loop_kwargs)

        # run the modified client loop
        client_loop_with_local_loader(
            args.client_id,
            dp_enabled=args.dp,
            noise_multiplier=args.noise,
            clip_norm=args.clip,
            debug=args.debug,
        )
    else:
        if sb_mgr:
            # Service Bus mode with full dataset (no partitioning)
            train_loader, _ = get_dataloader()
            run_client_loop_sb(
                train_loader,
                args.client_id,
                sb_manager=sb_mgr,
                dp_enabled=args.dp,
                noise_multiplier=args.noise,
                clip_norm=args.clip,
                debug=args.debug,
            )
        else:
            # Original HTTP mode
            client_loop(
                args.client_id,
                dp_enabled=args.dp,
                noise_multiplier=args.noise,
                clip_norm=args.clip,
                debug=args.debug,
            )