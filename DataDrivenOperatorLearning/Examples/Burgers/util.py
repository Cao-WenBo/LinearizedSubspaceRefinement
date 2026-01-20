import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Set
import networkx as nx

def find_closest_to_output(model: nn.Module) -> Dict:
    """
    Find parameters closest to the output layer in a neural network
    
    Args:
        model: Neural network model
    
    Returns:
        Dictionary containing parameter index information
    """
    # Build computational graph to analyze topological relationships
    def build_computational_graph():
        """Build a computational graph to analyze module dependencies"""
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add all modules as nodes
        for name, module in model.named_modules():
            G.add_node(name, module=module)
        
        # Add edges based on parent-child relationships
        for name, module in model.named_modules():
            for child_name, child_module in module.named_children():
                child_full_name = f"{name}.{child_name}" if name else child_name
                G.add_edge(name, child_full_name)
        
        return G
    
    # Build the computational graph
    G = build_computational_graph()
    
    # Perform topological sort to get execution order
    try:
        # Get topological order (forward pass order)
        topo_order = list(nx.topological_sort(G))
        
        # Last node in topological order is closest to output
        last_node_name = topo_order[-1]
        last_module = G.nodes[last_node_name]['module']
        
        # Find all parameters in the last module
        closest_params = []
        for param in last_module.parameters(recurse=False):
            closest_params.append(param)
        
        # If last module has no parameters, look for the nearest ancestor with parameters
        if not closest_params:
            # Find all ancestors of the last node
            ancestors = list(nx.ancestors(G, last_node_name))
            
            # Reverse topological order to start from closest to output
            for node_name in reversed(topo_order):
                if node_name in ancestors:
                    module = G.nodes[node_name]['module']
                    for param in module.parameters(recurse=False):
                        if param not in closest_params:
                            closest_params.append(param)
                    if closest_params:
                        break
        
    except nx.NetworkXUnfeasible:
        # Graph has cycles (e.g., recurrent networks)
        # Use a simpler approach for networks with cycles
        return find_closest_to_output_simple(model)
    
    # Build mapping from parameter to its name
    param_to_name = {}
    for name, param in model.named_parameters():
        param_to_name[param] = name
    
    # Find indices of closest parameters in the flattened vector
    current_idx = 0
    closest_indices = []
    closest_param_names = []
    
    for name, param in model.named_parameters():
        if param in closest_params:
            num_elements = param.numel()
            start_idx = current_idx
            end_idx = current_idx + num_elements - 1
            closest_indices.extend(range(start_idx, end_idx + 1))
            closest_param_names.append(name)
        current_idx += param.numel()
    
    # Prepare result
    result = {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'closest_param_names': closest_param_names,
        'closest_param_indices': closest_indices,
        'closest_param_count': len(closest_params),
    }
    
    return result


def find_closest_to_output_simple(model: nn.Module) -> Dict:
    """
    Simple fallback method for networks with cycles
    
    Args:
        model: Neural network model
    
    Returns:
        Dictionary containing parameter index information
    """
    # Get all parameters
    all_params = list(model.named_parameters())
    
    # Simple heuristic: parameters that appear in forward method
    # This is a simplified approach and may not work for all networks
    source_code = model.forward.__code__.co_names
    
    # Look for parameter names in forward method
    param_names_in_forward = []
    for name, _ in all_params:
        module_name = name.split('.')[0] if '.' in name else name
        if module_name in source_code:
            param_names_in_forward.append(name)
    
    # If we found parameters referenced in forward, use the last one
    if param_names_in_forward:
        # Get the last parameter referenced in forward
        last_param_name = param_names_in_forward[-1]
        
        # Find all parameters with this prefix
        closest_param_names = []
        for name, _ in all_params:
            if name.startswith(last_param_name.split('.')[0]):
                closest_param_names.append(name)
    else:
        # Fallback: use the last parameter in the list
        closest_param_names = [all_params[-1][0]]
    
    # Find indices of these parameters in flattened vector
    current_idx = 0
    closest_indices = []
    
    for name, param in model.named_parameters():
        if name in closest_param_names:
            num_elements = param.numel()
            start_idx = current_idx
            end_idx = current_idx + num_elements - 1
            closest_indices.extend(range(start_idx, end_idx + 1))
        current_idx += param.numel()
    
    return {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'closest_param_names': closest_param_names,
        'closest_param_indices': closest_indices,
        'closest_param_count': len(closest_param_names),
    }


# Fixed BranchedNet with proper padding
class BranchedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1)
        )
        self.branch2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(64, 10)  # Output layer
    
    def forward(self, x):
        x = self.conv1(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        # Both b1 and b2 should have the same spatial dimensions now
        x = torch.cat([b1, b2], dim=1)
        x = x.mean([2, 3])  # Global average pooling
        x = self.fc(x)
        return x


# Example 1: Simple MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 5)  # Output layer
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Example 3: Complex network with multiple output paths
class ComplexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(10, 20)
        self.path_a = nn.Sequential(
            nn.Linear(20, 30),
            nn.Linear(30, 10)
        )
        self.path_b = nn.Linear(20, 10)
        self.final = nn.Linear(20, 5)  # This is the output layer
    
    def forward(self, x):
        x = torch.relu(self.shared(x))
        a = self.path_a(x)
        b = self.path_b(x)
        combined = torch.cat([a, b], dim=1)
        return self.final(combined)


# Test SimpleMLP
print("Testing SimpleMLP:")
mlp_model = SimpleMLP()
mlp_result = find_closest_to_output(mlp_model)
print(f"Total parameters: {mlp_result['total_parameters']}")
print(f"Parameters closest to output: {mlp_result['closest_param_names']}")
print(f"Parameter index range: {min(mlp_result['closest_param_indices'])} to {max(mlp_result['closest_param_indices'])}")

# Test BranchedNet
print("\nTesting BranchedNet:")
branched_model = BranchedNet()
branched_result = find_closest_to_output(branched_model)
print(f"Total parameters: {branched_result['total_parameters']}")
print(f"Parameters closest to output: {branched_result['closest_param_names']}")
print(f"Parameter index range: {min(branched_result['closest_param_indices'])} to {max(branched_result['closest_param_indices'])}")

# Test ComplexNet
print("\nTesting ComplexNet:")
complex_model = ComplexNet()
complex_result = find_closest_to_output(complex_model)
print(f"Total parameters: {complex_result['total_parameters']}")
print(f"Parameters closest to output: {complex_result['closest_param_names']}")
print(f"Parameter index range: {min(complex_result['closest_param_indices'])} to {max(complex_result['closest_param_indices'])}")