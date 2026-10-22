import numpy as np
import torch
import torch.nn as nn

# 1. Load the data
data = np.load("demos.npz")
X = torch.tensor(data["X"])
Y = torch.tensor(data["Y"]).long()

# 2. Split into train (90%) and validation (10%)
torch.manual_seed(0)
perm = torch.randperm(len(X))
n_val = len(X) // 10
val_idx, train_idx = perm[:n_val], perm[n_val:]
X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

# 3. The network
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(344, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = Policy()

    # 4. Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training loop
    BATCH = 256
    for epoch in range(20):
        model.train()
        shuffle = torch.randperm(len(X_train))
        total_loss = 0.0

        for i in range(0, len(X_train), BATCH):
            idx = shuffle[i:i + BATCH]
            xb, yb = X_train[idx], Y_train[idx]

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            preds = model(X_val).argmax(dim=1)
            acc = (preds == Y_val).float().mean().item()

        print(f"epoch {epoch + 1:2d}  train loss {total_loss / len(X_train):.3f}  val acc {acc:.3f}")

    torch.save(model.state_dict(), "policy.pt")
    print("Saved policy.pt")