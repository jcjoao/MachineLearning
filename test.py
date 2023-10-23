import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# Load your data
x = np.load('Xtrain_Classification1.npy')
y = np.load('ytrain_Classification1.npy')

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Create custom dataset and data loaders
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Reshape the input to (3, 28, 28) assuming it's originally flattened
        x_item = self.x[idx].reshape(3, 28, 28)
        return torch.FloatTensor(x_item), torch.LongTensor([self.y[idx]])


batch_size = 64
train_dataset = CustomDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the model
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()  # Correct usage of super()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.fc1 = nn.Linear(16 * 26 * 26, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 16 * 26 * 26)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



model = CNNModel()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()

# Evaluation
# Evaluation
model.eval()
with torch.no_grad():
    # Initialize lists for true and predicted labels
    true_labels = []
    predicted_labels = []

    for inputs, labels in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())

    # Calculate balanced accuracy using scikit-learn
    balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)*100

print(f'Balanced Accuracy on the training data: {balanced_acc:.2f}')

# Now you can make predictions on new data