import torch
from fl_core.model_def import get_model
from copy import deepcopy

class Aggregator:
    def __init__(self):
        self.global_model = get_model()
        self.current_round = 0
        self.updates = []
        self.data_sizes = []
        self.last_num_updates = 0

    def get_weights(self):
        return {k: v.cpu().tolist() for k, v in self.global_model.state_dict().items()}

    @staticmethod
    def _to_tensor(val):
        """Convert a value to a tensor if it isn't already.

        Handles both JSON-deserialized lists (from HTTP) and torch.Tensor
        values (from Service Bus blob deserialization).
        """
        if isinstance(val, torch.Tensor):
            return val
        return torch.tensor(val)

    def receive_update(self, weights, data_size):
        self.updates.append(weights)
        self.data_sizes.append(data_size)

    def aggregate(self):
        if len(self.updates) == 0:
            return
        # We'll treat incoming 'weights' as updates/deltas (local - global).
        # Compute weighted average delta and add it to the current global model.
        global_state = deepcopy(self.global_model.state_dict())

        # Weighted averaging of deltas
        total_data = sum(self.data_sizes)

        # If for some reason total_data is zero (edge case), fall back to
        # uniform averaging across received updates to avoid division by zero.
        use_data_weights = total_data > 0

        avg_delta = {}
        for key in global_state.keys():
            if use_data_weights:
                avg_delta_tensor = sum(
                    (self._to_tensor(update[key]) * (ds / total_data))
                    for update, ds in zip(self.updates, self.data_sizes)
                )
            else:
                # uniform average over updates
                avg_delta_tensor = sum((self._to_tensor(update[key]) for update in self.updates)) / len(self.updates)
            avg_delta[key] = avg_delta_tensor

        # Apply averaged delta: global + avg_delta
        new_state = {k: (global_state[k].float() + avg_delta[k].float()) for k in global_state.keys()}

        self.global_model.load_state_dict(new_state)
        self.current_round += 1
        # record how many updates were used for this aggregation (diagnostics)
        self.last_num_updates = len(self.updates)
        self.updates = []
        self.data_sizes = []
