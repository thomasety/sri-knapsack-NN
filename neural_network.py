import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tabulation_data import generate_knapsack_data

class KnapsackNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(KnapsackNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),  
            nn.ReLU(),                  
            nn.Linear(64, 32),          
            nn.ReLU(),                  
            nn.Linear(32, output_size), 
            nn.Sigmoid()                
        )
    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    NUM_SAMPLES = 10000  
    NUM_ITEMS = 10      
    MAX_WEIGHT = 20
    MAX_VALUE = 100
    X_train, y_train = generate_knapsack_data(
        NUM_SAMPLES,
        max_weight=MAX_WEIGHT,
        max_value=MAX_VALUE
    )
    INPUT_SIZE = NUM_ITEMS*2+1 
    OUTPUT_SIZE = NUM_ITEMS  
    model = KnapsackNet(INPUT_SIZE, OUTPUT_SIZE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 64
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 2000
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")
    model.eval()


  
