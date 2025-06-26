import random
import torch 

def tabulation_knapsack(values, weights, capacity):
    n = len(values)
    tab = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                include_item = values[i-1] + tab[i-1][w-weights[i-1]]
                exclude_item = tab[i-1][w]
                tab[i][w] = max(include_item, exclude_item)
            else:
                tab[i][w] = tab[i-1][w]
    items_taken = [0] * n 
    w = capacity
    for i in range(n, 0, -1):
        if tab[i][w] == tab[i-1][w]:
            pass 
        else:
            items_taken[i-1] = 1 
            w -= weights[i-1]  
    return tab[n][capacity], items_taken

def generate_knapsack_data(num_samples, max_weight=20, max_value=100):
    all_features = []
    outputs = []
    for i in range(num_samples):
        num_items = 10
        values=[random.randint(1, max_value) for i in range(num_items)]
        weights = [random.randint(1, max_weight) for i in range(num_items)]
        capacity=50 

        i, items_taken = tabulation_knapsack(values, weights, capacity)
        normalized_values =[v/max_value for v in values]
        normalized_weights= [w/max_weight for w in weights]
        normalized_capacity = capacity/50

        features=normalized_values+normalized_weights+[normalized_capacity]
        all_features.append(features)
        outputs.append(items_taken)

    x = torch.tensor(all_features, dtype=torch.float32)
    y = torch.tensor(outputs, dtype=torch.float32)
    return x, y
print(generate_knapsack_data(1000))
