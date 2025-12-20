import torch
import copy

class ContributionEvaluator:
    def __init__(self, global_model, test_loader):
        self.base_model = global_model
        self.test_loader = test_loader

    def evaluate_utility(self, model):
        """Measures the utility (accuracy) of a specific model state."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        return correct / total

    def calculate_marginal_contribution(self, client_updates, current_global_weights):
        """
        Calculates how much each client's gradient improved the global model.
        This is a 'Cooperative Game' where we measure individual impact.
        """
        rewards = {}
        # all
        full_agg_weights = self._aggregate(client_updates, current_global_weights)
        full_utility = self.evaluate_utility(full_agg_weights)

        for client_id in client_updates:
           # leaving-one-out
            partial_updates = {k: v for k, v in client_updates.items() if k != client_id}
            partial_agg_weights = self._aggregate(partial_updates, current_global_weights)
            partial_utility = self.evaluate_utility(partial_agg_weights)
            
            rewards[client_id] = full_utility - partial_utility
            
        return rewards

    def _aggregate(self, updates, base_weights):
        """Simple FedAvg simulation for evaluation."""
        new_weights = copy.deepcopy(base_weights)
        if not updates: return new_weights
        
        for key in new_weights.keys():
            client_sum = torch.stack([u[key] for u in updates.values()]).mean(0)
            new_weights[key] = client_sum
        
        temp_model = copy.deepcopy(self.base_model)
        temp_model.load_state_dict(new_weights)
        return temp_model