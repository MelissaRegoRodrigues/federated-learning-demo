import torch
import copy
import numpy as np
from typing import Dict, Tuple

class ContributionEvaluator:
    """
    Avalia contribuição de clientes usando Monte Carlo Shapley Value.
    Shapley Value: mede a contribuição marginal média de cada jogador
    em todas as possíveis coalizões.
    """
    
    def __init__(self, global_model, test_loader, num_samples=50):
        """
        Args:
            global_model: modelo base
            test_loader: dataset de teste
            num_samples: amostras de coalizões aleatórias (Monte Carlo)
        """
        self.base_model = global_model
        self.test_loader = test_loader
        self.num_samples = num_samples
        self.device = next(global_model.parameters()).device

    def evaluate_utility(self, model):
        """Mede a acurácia (utilidade) de um modelo no conjunto de teste."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        return correct / total if total > 0 else 0.0

    def calculate_shapley_value(self, client_updates: Dict, current_global_weights) -> Dict:
        """
        Calcula Shapley Value para cada cliente usando Monte Carlo.
        
        Shapley Value:
        sv[i] = média de (U(S ∪ {i}) - U(S)) para TODAS as 2^n coalizões S
        
        Com n > 20 clientes, usamos Monte Carlo (amostragem de coalizões aleatórias).
        
        Args:
            client_updates: {client_id: state_dict}
            current_global_weights: pesos do modelo global
            
        Returns:
            {client_id: shapley_value}
        """
        n_clients = len(client_updates)
        client_ids = list(client_updates.keys())
        shapley_values = {cid: 0.0 for cid in client_ids}
        
        # Monte Carlo: amostrar coalizões aleatórias
        for sample in range(self.num_samples):
            # Gerar uma permutação aleatória dos clientes
            perm = np.random.permutation(client_ids)
            
            # Iterar pela permutação, computando contribuição incremental
            cumulative_set = set()
            for i, client_id in enumerate(perm):
                # Utilidade SEM este cliente
                utility_without = self._evaluate_coalition(cumulative_set, client_updates, current_global_weights)
                
                # Utilidade COM este cliente
                cumulative_set.add(client_id)
                utility_with = self._evaluate_coalition(cumulative_set, client_updates, current_global_weights)
                
                # Marginal contribution
                marginal = utility_with - utility_without
                shapley_values[client_id] += marginal / self.num_samples
        
        return shapley_values

    def _evaluate_coalition(self, coalition: set, client_updates: Dict, current_global_weights) -> float:
        """
        Avalia a utilidade de uma coalizão específica de clientes.
        
        Args:
            coalition: set de client_ids
            client_updates: todos os updates disponíveis
            current_global_weights: pesos base
            
        Returns:
            acurácia do modelo agregado dessa coalizão
        """
        if not coalition:
            # Coalizão vazia: usar modelo base
            base_copy = copy.deepcopy(self.base_model)
            base_copy.load_state_dict(current_global_weights)
            return self.evaluate_utility(base_copy)
        
        # Agregar updates da coalizão
        subset_updates = {cid: client_updates[cid] for cid in coalition}
        aggregated_model = self._aggregate(subset_updates, current_global_weights)
        return self.evaluate_utility(aggregated_model)

    def detect_free_riders(self, shapley_values: Dict, threshold_percentile=25) -> Tuple[set, Dict]:
        """
        Detecta free-riders com base em Shapley Value baixo.
        
        Args:
            shapley_values: {client_id: value}
            threshold_percentile: percentil abaixo do qual são considerados FR
            
        Returns:
            (free_rider_ids, client_scores)
        """
        values = np.array(list(shapley_values.values()))
        threshold = np.percentile(values, threshold_percentile)
        
        free_riders = {cid for cid, val in shapley_values.items() if val < threshold}
        
        return free_riders, shapley_values

    def _aggregate(self, updates, base_weights):
        """Agrega updates usando FedAvg simples."""
        new_weights = copy.deepcopy(base_weights)
        
        if not updates:
            return self.base_model
        
        for key in new_weights.keys():
            client_sum = torch.stack([u[key] for u in updates.values()]).mean(0)
            new_weights[key] = client_sum
        
        temp_model = copy.deepcopy(self.base_model)
        temp_model.load_state_dict(new_weights)
        return temp_model


class FreeRiderClient:
    """Simula um cliente com comportamento parasitário."""
    
    BEHAVIORS = {
        'honest': 'treina normalmente',
        'silent': 'envia gradiente zero',
        'noisy': 'envia gradiente com ruído alta',
        'lazy': 'treina com poucas épocas',
        'delayed': 'envia update atrasado'
    }
    
    def __init__(self, client_id: int, behavior: str = 'honest', malice_intensity: float = 0.5):
        """
        Args:
            client_id: identificador
            behavior: tipo de comportamento ('honest', 'silent', 'noisy', 'lazy', 'delayed')
            malice_intensity: intensidade da malícia [0, 1]
        """
        self.client_id = client_id
        self.behavior = behavior
        self.malice_intensity = malice_intensity
        self.is_free_rider = behavior != 'honest'
        assert behavior in self.BEHAVIORS, f"Comportamento {behavior} não suportado"

    def process_update(self, update_state_dict, epochs_trained: int):
        """
        Processa o update de acordo com o comportamento.
        
        Args:
            update_state_dict: gradientes/pesos locais
            epochs_trained: quantas épocas realmente treinou
            
        Returns:
            update_state_dict modificado (ou zero, ou com ruído)
        """
        if self.behavior == 'honest':
            return update_state_dict
        
        elif self.behavior == 'silent':
            # Retorna zeros
            return {k: torch.zeros_like(v) for k, v in update_state_dict.items()}
        
        elif self.behavior == 'noisy':
            # Adiciona ruído Gaussiano de alta variância
            noisy_update = copy.deepcopy(update_state_dict)
            for key in noisy_update:
                noise = torch.randn_like(noisy_update[key]) * self.malice_intensity * 10
                noisy_update[key] = noisy_update[key] + noise
            return noisy_update
        
        elif self.behavior == 'lazy':
            # Treina pouco → update pequeno (proporcional às épocas)
            lazy_update = copy.deepcopy(update_state_dict)
            factor = epochs_trained / 1.0  # Normalizar por 1 época esperada
            for key in lazy_update:
                lazy_update[key] = lazy_update[key] * min(factor, 0.3)  # Max 30% do update
            return lazy_update
        
        elif self.behavior == 'delayed':
            # Simula que enviou update desatualizado → scale down
            delayed_update = copy.deepcopy(update_state_dict)
            for key in delayed_update:
                delayed_update[key] = delayed_update[key] * (1.0 - self.malice_intensity)
            return delayed_update

    def __repr__(self):
        return f"Client_{self.client_id}({self.behavior})"
