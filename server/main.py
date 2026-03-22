# ============ IMPORTS ============
import os
import sys
from datetime import datetime
from math import ceil
import random

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.aggregator import Aggregator
from service_bus.config import ServiceBusConfig

# ============ AZURE SERVICE BUS SETUP ============
sb_config = ServiceBusConfig.from_env()
sb_manager = None

if sb_config.enabled:
    sb_config.validate()
    from service_bus.manager import ServiceBusManager
    sb_manager = ServiceBusManager(sb_config)
    print("[SERVER] Azure Service Bus transport ENABLED")

# ============ SETUP FASTAPI ============
app = FastAPI(title="Federated Learning Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ GLOBAL STATE ============
agg = Aggregator()
clients_status = {}  # Current status of each client
client_history = {}  # Full history per client: {client_id: [events]}
training_history = []  # Overall training metrics
round_participation = {}  # Track who was selected and who participated per round

# Configuration
PARTICIPATION_PROB = float(os.environ.get("FL_PARTICIPATION_PROB", 0.5))
AGGREGATION_TIMEOUT = float(os.environ.get("FL_AGG_TIMEOUT", 45.0))
MIN_UPDATES_TO_AGGREGATE = 2

# Track active clients dynamically
connected_clients = set()
first_submission_time = {}
selected_clients_by_round = {}
submitted_clients_by_round = {}
aggregation_in_progress = False  # Prevent double aggregation

# Adaptive participation
P_ADJUST_WINDOW = 5
P_IMPROVE_MIN = 1e-3
P_INCREMENT = 0.1
last_p_adjust_round = -1


def get_num_clients():
    """Get number of connected clients dynamically"""
    return len(connected_clients) if len(connected_clients) > 0 else MIN_UPDATES_TO_AGGREGATE


def get_updates_per_round():
    """Calculate target updates based on connected clients"""
    num_clients = get_num_clients()
    target = max(MIN_UPDATES_TO_AGGREGATE, ceil(PARTICIPATION_PROB * num_clients))
    return min(target, num_clients)


def add_to_client_history(client_id, event_type, details=None):
    """Add event to client's history"""
    if client_id not in client_history:
        client_history[client_id] = []
    
    event = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        "round": agg.current_round,
        "details": details or {}
    }
    client_history[client_id].append(event)


def select_participants_for_round(round_idx: int):
    """Select participants from connected clients"""
    if round_idx in selected_clients_by_round:
        return selected_clients_by_round[round_idx]

    if len(connected_clients) == 0:
        return set()

    num_clients = len(connected_clients)
    k = max(1, min(get_updates_per_round(), num_clients))
    
    sampled = set(random.sample(list(connected_clients), k))
    selected_clients_by_round[round_idx] = sampled
    first_submission_time.setdefault(round_idx, time.time())
    
    # Initialize round participation tracking
    round_participation[round_idx] = {
        "selected": list(sampled),
        "submitted": [],
        "timed_out": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Add selection to history for selected clients
    for client_id in sampled:
        add_to_client_history(client_id, "selected", {"round": round_idx})
    
    print(f"[SERVER] Selected {len(sampled)} participants for round {round_idx}: {sorted(list(sampled))}")

    # Notify clients via Service Bus (if enabled)
    if sb_manager:
        try:
            sb_manager.publish_round_control(round_idx, sorted(list(sampled)))
        except Exception as e:
            print(f"[SB] Failed to publish round-control: {e}")

    return sampled


def adjust_participation():
    """Increase PARTICIPATION_PROB when convergence stalls"""
    global PARTICIPATION_PROB, last_p_adjust_round
    current = agg.current_round
    
    if current <= 0 or last_p_adjust_round >= current:
        return
    if len(training_history) < P_ADJUST_WINDOW:
        return

    recent = training_history[-P_ADJUST_WINDOW:]
    first_acc = recent[0]["accuracy"]
    last_acc = recent[-1]["accuracy"]
    improvement = last_acc - first_acc
    
    if improvement < P_IMPROVE_MIN and PARTICIPATION_PROB < 1.0:
        PARTICIPATION_PROB = min(1.0, PARTICIPATION_PROB + P_INCREMENT)
        last_p_adjust_round = current
        print(f"[SERVER] ↑ Participation increased to {PARTICIPATION_PROB:.2f}")


def perform_aggregation(round_num, trigger="normal"):
    """Perform aggregation and prevent duplicates"""
    global aggregation_in_progress
    
    if aggregation_in_progress:
        print(f"[SERVER] Aggregation already in progress, skipping")
        return False
    
    aggregation_in_progress = True
    
    try:
        print(f"[SERVER] 🔄 Aggregating round {round_num}... (trigger: {trigger})")
        agg.aggregate()
        
        finished_round = agg.current_round - 1
        
        # Update round participation tracking
        if round_num in round_participation:
            submitted_set = submitted_clients_by_round.get(round_num, set())
            round_participation[round_num]["submitted"] = list(submitted_set)
        
        # Cleanup
        if finished_round in submitted_clients_by_round:
            del submitted_clients_by_round[finished_round]
        if finished_round in first_submission_time:
            del first_submission_time[finished_round]

        try:
            adjust_participation()
        except Exception as e:
            print(f"[SERVER] Participation adjustment failed: {e}")

        accuracy, loss = evaluate_model()
        
        # Get participation stats for this round
        participated = round_participation.get(finished_round, {})
        
        training_history.append({
            "round": finished_round,
            "accuracy": accuracy,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
            "num_updates": agg.last_num_updates,
            "selected": participated.get("selected", []),
            "submitted": participated.get("submitted", []),
            "timed_out": participated.get("timed_out", [])
        })

        print(f"[SERVER] ✅ Round {finished_round} complete. Acc: {accuracy:.2%}, Loss: {loss:.4f}")

        # Publish via Service Bus (if enabled)
        if sb_manager:
            try:
                sb_manager.publish_global_model(
                    agg.current_round, agg.global_model.state_dict()
                )
                sb_manager.publish_dashboard_event({
                    "type": "round_complete",
                    "round": finished_round,
                    "accuracy": accuracy,
                    "loss": loss,
                    "num_updates": agg.last_num_updates,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                print(f"[SB] Failed to publish post-aggregation events: {e}")

        return True

    finally:
        aggregation_in_progress = False


# ============ DATA VALIDATION ============
class UpdateRequest(BaseModel):
    client_id: str
    weights: dict
    data_size: int
    # client-provided round id to prevent stale updates
    round: int


# ============ API ENDPOINTS ============

@app.get("/")
def root():
    return {
        "status": "Federated Learning Server Running",
        "round": agg.current_round,
        "connected_clients": len(connected_clients),
        "pending_updates": len(agg.updates),
    }


@app.get("/get_model")
def get_model():
    return {
        "round": agg.current_round,
        "weights": agg.get_weights(),
    }


@app.get("/should_participate")
def should_participate(client_id: str):
    """Check if client should participate and mark as connected"""
    connected_clients.add(client_id)
    
    # Check if selected for current round
    selected_set = select_participants_for_round(agg.current_round)
    selected = str(client_id) in selected_set
    
    # Update status
    if selected:
        clients_status[client_id] = {
            "status": "selected",
            "timestamp": datetime.now().isoformat(),
            "round": agg.current_round,
        }
        add_to_client_history(client_id, "waiting_to_train", {"round": agg.current_round})
    else:
        clients_status[client_id] = {
            "status": "waiting",
            "timestamp": datetime.now().isoformat(),
            "round": agg.current_round,
        }
    
    return {"selected": selected, "round": agg.current_round}


@app.post("/submit_update")
def submit_update(upd: UpdateRequest):
    # Receive a client update and validate the client-provided round id.
    current_round = agg.current_round

    if upd.round != current_round:
        print(f"[SERVER] ✗ Stale/future submission from client {upd.client_id} (client_round={upd.round}, server_round={current_round}) - rejecting")
        # HTTP 409 Conflict is appropriate for stale submissions; return descriptive body
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail={"status": "stale_round", "client_round": upd.round, "server_round": current_round})

    # Now proceed with the normal submission for the current round
    submitted_set = submitted_clients_by_round.get(current_round, set())

    connected_clients.add(upd.client_id)

    if upd.client_id in submitted_set:
        print(f"[SERVER] ✗ Duplicate submission from client {upd.client_id} - rejecting")
        return {"status": "duplicate", "round": current_round}

    # Accept update
    try:
        agg.receive_update(upd.weights, upd.data_size)
    except Exception as e:
        print(f"[SERVER] ✗ Failed to process update from {upd.client_id}: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail={"status": "invalid_update", "error": str(e)})

    submitted_set.add(upd.client_id)
    submitted_clients_by_round[current_round] = submitted_set

    # Update status and history
    clients_status[upd.client_id] = {
        "status": "submitted",
        "timestamp": datetime.now().isoformat(),
        "data_size": upd.data_size,
        "round": current_round,
    }

    add_to_client_history(upd.client_id, "submitted", {
        "round": current_round,
        "data_size": upd.data_size
    })

    updates_needed = get_updates_per_round()
    print(f"[SERVER] ✓ Client {upd.client_id} submitted. Updates: {len(agg.updates)}/{updates_needed}")

    if len(agg.updates) >= updates_needed:
        success = perform_aggregation(current_round, trigger="sufficient_updates")

        if success:
            return {
                "status": "aggregated",
                "round": agg.current_round,
            }

    return {
        "status": "received",
        "round": agg.current_round,
        "pending_updates": len(agg.updates),
    }


@app.get("/status")
def get_status():
    """Return status for connected clients only"""
    return {
        "current_round": agg.current_round,
        "active_clients": len([c for c in clients_status.values() if c.get("status") in ["submitted", "selected", "waiting"]]),
        "pending_updates": len(agg.updates),
        "clients": clients_status,
        "history": training_history[-20:],
        "connected_clients": list(connected_clients),
        "aggregation_in_progress": aggregation_in_progress,
    }


@app.get("/metrics")
def get_metrics():
    """Return training metrics - only updated after aggregation"""
    return {
        "round": agg.current_round,
        "rounds": [h["round"] for h in training_history],
        "accuracy": [h["accuracy"] for h in training_history],
        "loss": [h["loss"] for h in training_history],
        "last_updated": training_history[-1]["timestamp"] if training_history else None
    }


@app.get("/client_history/{client_id}")
def get_client_history(client_id: str):
    """Get full history for a specific client"""
    return {
        "client_id": client_id,
        "history": client_history.get(client_id, []),
        "current_status": clients_status.get(client_id, {}),
        "total_events": len(client_history.get(client_id, []))
    }


@app.get("/round_details/{round_num}")
def get_round_details(round_num: int):
    """Get detailed information about a specific round"""
    round_data = next((h for h in training_history if h["round"] == round_num), None)
    participation = round_participation.get(round_num, {})
    
    if round_data:
        return {
            "round": round_num,
            "accuracy": round_data["accuracy"],
            "loss": round_data["loss"],
            "timestamp": round_data["timestamp"],
            "selected_clients": participation.get("selected", []),
            "submitted_clients": participation.get("submitted", []),
            "timed_out_clients": participation.get("timed_out", []),
            "participation_rate": len(participation.get("submitted", [])) / max(len(participation.get("selected", [])), 1)
        }
    
    return {"error": "Round not found"}


@app.get("/config")
def get_config():
    """Return dynamic configuration"""
    num_clients = get_num_clients()
    updates_per_round = get_updates_per_round()
    
    return {
        "num_clients": num_clients,
        "participation_prob": PARTICIPATION_PROB,
        "updates_per_round": updates_per_round,
        "aggregation_timeout": AGGREGATION_TIMEOUT,
        "current_round": agg.current_round,
        "connected_clients": len(connected_clients),
    }


# Background monitor for timeouts
def _aggregation_monitor():
    while True:
        try:
            current = agg.current_round
            start_ts = first_submission_time.get(current, None)
            
            if start_ts is not None and not aggregation_in_progress:
                elapsed = time.time() - start_ts
                if elapsed >= AGGREGATION_TIMEOUT:
                    had_updates = len(agg.updates) > 0
                    
                    # Mark missing clients as timed_out
                    selected = selected_clients_by_round.get(current, set())
                    submitted = submitted_clients_by_round.get(current, set())
                    missing = selected - submitted
                    
                    for cid in missing:
                        clients_status[cid] = {
                            "status": "timed_out",
                            "timestamp": datetime.now().isoformat(),
                            "round": current,
                        }
                        add_to_client_history(cid, "timed_out", {"round": current})
                    
                    # Update round participation
                    if current in round_participation:
                        round_participation[current]["timed_out"] = list(missing)

                    if had_updates:
                        print(f"[SERVER] Timeout ({AGGREGATION_TIMEOUT}s) - aggregating {len(agg.updates)} updates")
                        perform_aggregation(current, trigger="timeout")
                    else:
                        print(f"[SERVER] Timeout with no updates - advancing round")
                        agg.current_round += 1
                        
                        # Cleanup
                        if current in submitted_clients_by_round:
                            del submitted_clients_by_round[current]
                        if current in first_submission_time:
                            del first_submission_time[current]

            time.sleep(1.0)
        except Exception as e:
            print(f"[SERVER] Monitor error: {e}")
            time.sleep(1.0)


monitor_thread = threading.Thread(target=_aggregation_monitor, daemon=True)
monitor_thread.start()


# ============ SERVICE BUS RECEIVER THREAD ============

def _sb_update_receiver():
    """Background thread: receive client updates from the Service Bus queue.

    Mirrors the logic in the /submit_update endpoint but reads from the
    'client-updates' queue instead of HTTP POST. Uses the claim-check
    pattern to retrieve full weight tensors from Blob Storage.
    """
    if not sb_manager:
        return

    def handle_update(client_id, round_id, weights, data_size):
        current_round = agg.current_round

        # Round validation (same as HTTP endpoint)
        if round_id != current_round:
            print(f"[SB] Stale update from client {client_id} "
                  f"(client_round={round_id}, server_round={current_round}) - rejecting")
            return

        # Duplicate check
        submitted_set = submitted_clients_by_round.get(current_round, set())
        if client_id in submitted_set:
            print(f"[SB] Duplicate submission from client {client_id} - rejecting")
            return

        connected_clients.add(client_id)

        # Weights arrive as {str: Tensor} from blob deserialization;
        # convert to JSON-compatible lists so the aggregator handles them
        # the same way as HTTP submissions.
        weights_as_lists = {k: v.cpu().tolist() for k, v in weights.items()}

        try:
            agg.receive_update(weights_as_lists, data_size)
        except Exception as e:
            print(f"[SB] Failed to process update from {client_id}: {e}")
            return

        submitted_set.add(client_id)
        submitted_clients_by_round[current_round] = submitted_set

        clients_status[client_id] = {
            "status": "submitted",
            "timestamp": datetime.now().isoformat(),
            "data_size": data_size,
            "round": current_round,
        }
        add_to_client_history(client_id, "submitted", {
            "round": current_round,
            "data_size": data_size,
            "transport": "servicebus",
        })

        updates_needed = get_updates_per_round()
        print(f"[SB] Client {client_id} submitted. Updates: {len(agg.updates)}/{updates_needed}")

        if len(agg.updates) >= updates_needed:
            perform_aggregation(current_round, trigger="sufficient_updates_sb")

    print("[SB] Starting client-updates receiver thread...")
    sb_manager.receive_client_updates(handle_update)


if sb_manager:
    sb_receiver_thread = threading.Thread(target=_sb_update_receiver, daemon=True)
    sb_receiver_thread.start()


# ============ HELPER FUNCTIONS ============

def evaluate_model():
    """Evaluate global model on test set"""
    import torch.nn.functional as F
    from clients.data_utils import get_dataloader

    _, test_loader = get_dataloader(batch_size=128)
    model = agg.global_model
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            num_batches += 1
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / num_batches
    return accuracy, avg_loss


# ============ START SERVER ============
if __name__ == "__main__":
    import uvicorn

    transport_mode = "Azure Service Bus" if sb_manager else "HTTP (REST)"

    print("=" * 70)
    print("🚀 FEDERATED LEARNING SERVER")
    print("=" * 70)
    print(f"📍 Server:    http://127.0.0.1:9000")
    print(f"🔗 Transport: {transport_mode}")
    print(f"⚙️  Min Updates: {MIN_UPDATES_TO_AGGREGATE}")
    print(f"⚙️  Participation: {PARTICIPATION_PROB * 100:.0f}%")
    print(f"⚙️  Timeout: {AGGREGATION_TIMEOUT}s")
    print("=" * 70)
    print("📱 Dynamic client tracking with history")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=9000)