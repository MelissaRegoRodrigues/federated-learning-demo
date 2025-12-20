import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models.sample_cnn import SimpleCNN
from src.utils.data_manager import partition_data
from src.game_theory.incentive_manager import ContributionEvaluator

def train_local_client(model, data_loader, epochs=1):
    """Simulates local training on a client device."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=config['model_params']['batch_size'])

    client_indices = partition_data(
        train_set.targets.numpy(), 
        config['federated_params']['num_clients'], 
        alpha=config['federated_params']['alpha']
    )
    
    global_model = SimpleCNN().to(device)
    evaluator = ContributionEvaluator(global_model, test_loader)

    for r in range(config['federated_params']['rounds']):
        print(f"\n--- Round {r+1} ---")
        client_updates = {}

        for i in range(config['federated_params']['num_clients']):
            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            client_loader = DataLoader(Subset(train_set, client_indices[i]), batch_size=32, shuffle=True)
          
            print(f"Client {i} training on {len(client_indices[i])} samples...")
            client_updates[i] = train_local_client(local_model, client_loader)


        # We use the evaluator to see how much each update is worth
        print("Calculating Shapley Values (Incentive Distribution)...")
        contributions = evaluator.calculate_marginal_contribution(
            client_updates, 
            global_model.state_dict()
        )

        # Instead of simple mean, we weight by the game-theory 'contribution' score
        new_global_weights = {}
        total_contribution = sum(contributions.values())
        
        for key in global_model.state_dict().keys():
            weighted_sum = torch.zeros_like(global_model.state_dict()[key])
            for i, update in client_updates.items():
                # Normalize weight: contribution of client i / total contributions
                weight = contributions[i] / (total_contribution + 1e-9) 
                weighted_sum += update[key] * weight
            new_global_weights[key] = weighted_sum
        
        global_model.load_state_dict(new_global_weights)

        accuracy = evaluator.evaluate_utility(global_model)
        print(f"Round {r+1} Global Accuracy: {accuracy:.4f}")
        for i, val in contributions.items():
            print(f" > Client {i} Shapley Value: {val:.4f}")

if __name__ == "__main__":
    main()