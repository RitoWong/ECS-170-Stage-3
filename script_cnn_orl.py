import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load ORL dataset

with open("ORL", "rb") as f:
    data = pickle.load(f)

X_train = []
y_train = []

for instance in data["train"]:
    image = instance["image"]

    # ORL is stored as 112 x 92 x 3, but RGB channels are identical.
    # Use only one channel.
    image = image[:, :, 0]

    X_train.append(image)
    y_train.append(instance["label"] - 1)  # labels 1-40 -> 0-39

X_test = []
y_test = []

for instance in data["test"]:
    image = instance["image"]
    image = image[:, :, 0]

    X_test.append(image)
    y_test.append(instance["label"] - 1)

# 2. Convert to tensors
# ORL shape: N x 1 x 112 x 92

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
y_test = torch.tensor(y_test, dtype=torch.long)

# 3. DataLoader

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. CNN Model for ORL

class CNN_ORL(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 112x92 -> 56x46

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 56x46 -> 28x23

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)    # 28x23 -> 14x11
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 11, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 40)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = CNN_ORL()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Train

epochs = 30
loss_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# 6. Test

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.numpy())
        all_labels.extend(y_batch.numpy())


accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print("\n************ ORL CNN Performance ************")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# 7. Save loss curve

plt.figure()
plt.plot(range(1, epochs + 1), loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("ORL CNN Training Loss Curve")
plt.grid(True)
plt.savefig("orl_cnn_loss_curve.png")
plt.close()
