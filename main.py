"""
MVP: Validação Empírica de Mecanismos de Incentivo em FL
Compara FedAvg puro vs. FedAvg com Shapley Value

Uso:
    python main.py --num_clients 5 --rounds 10 --free_riders 2
"""

import torch
import yaml
import json
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime
import copy

from src.models.sample_cnn import SimpleCNN
from src.utils.data_manager import partition_data
from src.game_theory.incentive_manager import ContributionEvaluator, FreeRiderClient


def train_local_client(model, data_loader, device, epochs=1):
    """Treina cliente localmente."""
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


def run_fl_experiment(
    strategy: str,  # 'fedavg' ou 'game_theory'
    num_clients: int,
    num_rounds: int,
    train_set,
    test_loader,
    client_indices,
    device,
    free_rider_configs=None,
    shapley_samples=50
):
    """
    Roda um experimento completo de FL.
    
    Args:
        strategy: 'fedavg' (baseline) ou 'game_theory' (com Shapley)
        num_clients: número de clientes
        num_rounds: número de rodadas FL
        train_set: dataset de treino
        test_loader: dataset de teste
        client_indices: índices particionados
        device: cuda/cpu
        free_rider_configs: {client_id: ('behavior', intensity)} ou None
        shapley_samples: amostras para Monte Carlo Shapley
    
    Returns:
        resultados: {round: metrics}
    """
    
    global_model = SimpleCNN().to(device)
    evaluator = ContributionEvaluator(global_model, test_loader, num_samples=shapley_samples)
    
    results = {
        'strategy': strategy,
        'num_clients': num_clients,
        'free_riders': len(free_rider_configs) if free_rider_configs else 0,
        'rounds': []
    }
    
    if free_rider_configs is None:
        free_rider_configs = {}
    
    # Instanciar clientes com comportamentos
    clients = {}
    for i in range(num_clients):
        if i in free_rider_configs:
            behavior, intensity = free_rider_configs[i]
            clients[i] = FreeRiderClient(i, behavior=behavior, malice_intensity=intensity)
        else:
            clients[i] = FreeRiderClient(i, behavior='honest')
    
    print(f"\n{'='*60}")
    print(f"Experimento: {strategy.upper()}")
    print(f"Clientes: {num_clients} | Free-riders: {len(free_rider_configs)} | Rodadas: {num_rounds}")
    print(f"Comportamentos: {[(i, c.behavior) for i, c in clients.items() if c.is_free_rider]}")
    print(f"{'='*60}\n")
    
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        client_updates = {}
        client_shapley = {}
        detected_free_riders = set()
        
        # Treino local
        for i in range(num_clients):
            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            client_loader = DataLoader(
                Subset(train_set, client_indices[i]),
                batch_size=32,
                shuffle=True
            )
            
            # Treinar
            update = train_local_client(local_model, client_loader, device, epochs=1)
            
            # Aplicar comportamento malicioso (se aplicável)
            update = clients[i].process_update(update, epochs_trained=1)
            client_updates[i] = update
        
        # Agregação
        if strategy == 'fedavg':
            # FedAvg puro: média aritmética
            new_global_weights = {}
            for key in global_model.state_dict().keys():
                client_sum = torch.stack([u[key] for u in client_updates.values()]).mean(0)
                new_global_weights[key] = client_sum
        
        elif strategy == 'game_theory':
            # Com Shapley Value
            shapley_values = evaluator.calculate_shapley_value(
                client_updates,
                global_model.state_dict()
            )
            
            detected_free_riders, client_shapley = evaluator.detect_free_riders(
                shapley_values,
                threshold_percentile=25
            )
            
            # Normalizar pesos por contribuição (Shapley)
            min_contrib = min(shapley_values.values())
            normalized_contrib = {
                k: max(0, v - min_contrib) for k, v in shapley_values.items()
            }
            total_contrib = sum(normalized_contrib.values()) + 1e-9
            
            new_global_weights = {}
            for key in global_model.state_dict().keys():
                weighted_sum = torch.zeros_like(global_model.state_dict()[key])
                for i in range(num_clients):
                    weight = normalized_contrib[i] / total_contrib
                    weighted_sum += client_updates[i][key] * weight
                new_global_weights[key] = weighted_sum
        
        # Atualizar modelo global
        global_model.load_state_dict(new_global_weights)
        
        # Avaliar
        accuracy = evaluator.evaluate_utility(global_model)
        
        # Registrar resultados desta rodada
        round_result = {
            'round': round_num + 1,
            'accuracy': float(accuracy),
            'detected_free_riders': list(detected_free_riders),
            'num_detected': len(detected_free_riders)
        }
        
        if strategy == 'game_theory':
            round_result['shapley_values'] = {str(k): float(v) for k, v in client_shapley.items()}
        
        results['rounds'].append(round_result)
        
        print(f"  Acurácia: {accuracy:.4f} | Detected FR: {len(detected_free_riders)}/{len(free_rider_configs)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MVP: FL com Mecanismos de Incentivo")
    parser.add_argument('--num_clients', type=int, default=5, help='Número de clientes')
    parser.add_argument('--rounds', type=int, default=10, help='Rodadas de FL')
    parser.add_argument('--free_riders', type=int, default=2, help='Número de free-riders')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--shapley_samples', type=int, default=50, help='Amostras Monte Carlo Shapley')
    
    args = parser.parse_args()
    
    # Fixar seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}\n")
    
    # Carregar dados
    print("Carregando MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64)
    
    # Particionar dados (Non-IID com Dirichlet)
    print(f"Particionando {len(train_set)} amostras em {args.num_clients} clientes (alpha=0.5, Non-IID)...")
    client_indices = partition_data(
        train_set.targets.numpy(),
        args.num_clients,
        alpha=0.5
    )
    
    # Configurar free-riders
    free_rider_configs = {}
    behaviors = ['silent', 'noisy', 'lazy']
    for i in range(args.free_riders):
        behavior = behaviors[i % len(behaviors)]
        free_rider_configs[i] = (behavior, 0.7)  # intensidade alta
    
    # Rodar experimentos
    results_all = []
    
    # Experimento 1: FedAvg puro (baseline)
    result_fedavg = run_fl_experiment(
        strategy='fedavg',
        num_clients=args.num_clients,
        num_rounds=args.rounds,
        train_set=train_set,
        test_loader=test_loader,
        client_indices=client_indices,
        device=device,
        free_rider_configs=free_rider_configs,
        shapley_samples=args.shapley_samples
    )
    results_all.append(result_fedavg)
    
    # Experimento 2: Com Shapley Value
    result_game_theory = run_fl_experiment(
        strategy='game_theory',
        num_clients=args.num_clients,
        num_rounds=args.rounds,
        train_set=train_set,
        test_loader=test_loader,
        client_indices=client_indices,
        device=device,
        free_rider_configs=free_rider_configs,
        shapley_samples=args.shapley_samples
    )
    results_all.append(result_game_theory)
    
    # Salvar resultados
    output_dir = Path('./results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = output_dir / f'results_{timestamp}.json'
    
    with open(result_file, 'w') as f:
        json.dump(results_all, f, indent=2)
    
    print(f"\n✅ Resultados salvos em: {result_file}\n")
    
    # Resumo comparativo
    print("\n" + "="*60)
    print("RESUMO COMPARATIVO")
    print("="*60)
    
    acc_fedavg = [r['accuracy'] for r in result_fedavg['rounds']]
    acc_game_theory = [r['accuracy'] for r in result_game_theory['rounds']]
    
    print(f"FedAvg (Baseline):")
    print(f"  Acurácia inicial: {acc_fedavg[0]:.4f}")
    print(f"  Acurácia final:   {acc_fedavg[-1]:.4f}")
    print(f"  Melhoria:         {(acc_fedavg[-1] - acc_fedavg[0])*100:.2f}%")
    
    print(f"\nCom Shapley Value (Proposta):")
    print(f"  Acurácia inicial: {acc_game_theory[0]:.4f}")
    print(f"  Acurácia final:   {acc_game_theory[-1]:.4f}")
    print(f"  Melhoria:         {(acc_game_theory[-1] - acc_game_theory[0])*100:.2f}%")
    
    avg_detected = np.mean([r['num_detected'] for r in result_game_theory['rounds']])
    print(f"\nDetecção de Free-Riders (Game Theory):")
    print(f"  Média por rodada: {avg_detected:.1f}/{args.free_riders}")
    print(f"  Taxa de acerto:   {(avg_detected / args.free_riders) * 100:.1f}%")
    
    print(f"\nVantagem final (Shapley vs. FedAvg): {(acc_game_theory[-1] - acc_fedavg[-1])*100:+.2f}%")
    print("="*60 + "\n")
    
    # Gerar gráficos
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        rounds = [r['round'] for r in result_fedavg['rounds']]
        
        plt.plot(rounds, acc_fedavg, 'o-', label='FedAvg (Baseline)', linewidth=2, markersize=6)
        plt.plot(rounds, acc_game_theory, 's-', label='FedAvg + Shapley Value', linewidth=2, markersize=6)
        
        plt.xlabel('Rodada', fontsize=12)
        plt.ylabel('Acurácia Global', fontsize=12)
        plt.title(f'Convergência: {args.num_clients} clientes, {args.free_riders} free-riders', fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = output_dir / f'convergence_{timestamp}.png'
        plt.savefig(plot_file, dpi=100)
        print(f"✅ Gráfico salvo em: {plot_file}\n")
        plt.close()
    
    except Exception as e:
        print(f"⚠️  Não foi possível gerar gráfico: {e}\n")


if __name__ == "__main__":
    main()
