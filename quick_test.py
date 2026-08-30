import torch
import numpy as np
from pathlib import Path
import json

from src.models.sample_cnn import SimpleCNN
from src.game_theory.incentive_manager import ContributionEvaluator, FreeRiderClient

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")
print(f"Device: {device}\n")

print("Criando dataset sintético...")

# 100 amostras de treino, 50 de teste (imagens 1x28x28 MNIST-like)
n_train, n_test = 100, 50
train_images = torch.randn(n_train, 1, 28, 28)
train_labels = torch.randint(0, 10, (n_train,))
test_images = torch.randn(n_test, 1, 28, 28)
test_labels = torch.randint(0, 10, (n_test,))

from torch.utils.data import TensorDataset, DataLoader

train_set = TensorDataset(train_images, train_labels)
test_set = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_set, batch_size=32)

# Particionar entre 3 clientes
client_indices = [
    list(range(0, 35)),
    list(range(35, 70)),   
    list(range(70, 100))   
]

def train_local_client(model, data_loader, device, epochs=1):
    """Treina localmente."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# EXPERIMENTO
print("\n" + "="*60)
print("TESTE RÁPIDO: FedAvg vs. Shapley Value com Free-Riders")
print("="*60 + "\n")

results = {'fedavg': [], 'shapley': [], 'shapley_detections': []}

for strategy in ['fedavg', 'shapley']:
    print(f"\n{strategy.upper()}")
    print("-" * 40)
    
    global_model = SimpleCNN().to(device)
    evaluator = ContributionEvaluator(global_model, test_loader, num_samples=5)
    
    # Client 0 é silent free-rider
    clients = {
        0: FreeRiderClient(0, behavior='silent', malice_intensity=1.0),
        1: FreeRiderClient(1, behavior='honest'),
        2: FreeRiderClient(2, behavior='honest')
    }
    
    for round_num in range(3):  
        print(f"  Rodada {round_num + 1}/3", end="  ")
        
        client_updates = {}
        
        # Treinar localmente
        for i in range(3):
            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            client_subset = TensorDataset(
                train_images[client_indices[i]],
                train_labels[client_indices[i]]
            )
            client_loader = DataLoader(client_subset, batch_size=16, shuffle=True)
            
            update = train_local_client(local_model, client_loader, device, epochs=1)
            update = clients[i].process_update(update, epochs_trained=1)
            client_updates[i] = update
        
        # Agregar
        if strategy == 'fedavg':
            # Média simples
            new_global_weights = {}
            for key in global_model.state_dict().keys():
                client_sum = torch.stack([u[key] for u in client_updates.values()]).mean(0)
                new_global_weights[key] = client_sum
            num_detected = 0
        
        else:  # shapley
            shapley_values = evaluator.calculate_shapley_value(
                client_updates,
                global_model.state_dict()
            )
            
            detected_fr, _ = evaluator.detect_free_riders(
                shapley_values,
                threshold_percentile=25
            )
            num_detected = len(detected_fr)
            
            # Normalizar pesos
            min_contrib = min(shapley_values.values())
            normalized_contrib = {k: max(0, v - min_contrib) for k, v in shapley_values.items()}
            total_contrib = sum(normalized_contrib.values()) + 1e-9
            
            new_global_weights = {}
            for key in global_model.state_dict().keys():
                weighted_sum = torch.zeros_like(global_model.state_dict()[key])
                for i in range(3):
                    weight = normalized_contrib[i] / total_contrib
                    weighted_sum += client_updates[i][key] * weight
                new_global_weights[key] = weighted_sum
        
        global_model.load_state_dict(new_global_weights)
        accuracy = evaluator.evaluate_utility(global_model)
        results[strategy].append(accuracy)
        
        if strategy == 'shapley':
            results['shapley_detections'].append(num_detected)
            print(f"Acurácia: {accuracy:.4f} | FR Detectados: {num_detected}/1")
        else:
            print(f"Acurácia: {accuracy:.4f}")

# RESUMO
print("\n" + "="*60)
print("RESULTADOS FINAIS")
print("="*60)
print(f"FedAvg (Baseline):      {results['fedavg'][-1]:.4f}")
print(f"Shapley (Proposta):     {results['shapley'][-1]:.4f}")
print(f"Diferença:              {(results['shapley'][-1] - results['fedavg'][-1])*100:+.2f}%")
print(f"\nDetecção de Free-Riders (Shapley):")
print(f"  Acertos: {sum(results['shapley_detections'])} / {len(results['shapley_detections'])} rodadas")
print("="*60 + "\n")

# Salvar
output_dir = Path('./results')
output_dir.mkdir(exist_ok=True)
result_file = output_dir / 'quick_test_results.json'
with open(result_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Resultados salvos em: {result_file}\n")

# Gerar gráfico rápido
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    rounds = [1, 2, 3]
    plt.plot(rounds, results['fedavg'], 'o-', label='FedAvg (Baseline)', linewidth=2)
    plt.plot(rounds, results['shapley'], 's-', label='FedAvg + Shapley Value', linewidth=2)
    plt.xlabel('Rodada')
    plt.ylabel('Acurácia')
    plt.title('Teste Rápido: Convergência com Free-Riders')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_file = output_dir / 'quick_test_convergence.png'
    plt.savefig(plot_file, dpi=100)
    print(f"✅ Gráfico salvo em: {plot_file}\n")
    plt.close()
except Exception as e:
    print(f"⚠️  Matplotlib não disponível: {e}\n")
