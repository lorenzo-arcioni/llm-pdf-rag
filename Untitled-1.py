# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# %%
# -------------------------
# LWLR con SGD
# -------------------------
def lwlr_SGD(x_q, X, y, t=0.5, lr=0.01, epochs=50):
    m, d = X.shape
    theta = torch.zeros(d, 1, requires_grad=True)
    bias  = torch.zeros(1, requires_grad=True)

    for _ in range(epochs):
        for i in range(m):
            xi = X[i].unsqueeze(0)  # (1,d)
            yi = y[i]               # scalare

            diff = xi - x_q
            wi   = torch.exp(- (diff.norm()**2) / (2 * t ** 2))

            y_pred = (xi @ theta + bias).squeeze()

            loss = 0.5 * wi * (yi - y_pred)**2
            loss.backward()

            with torch.no_grad():
                theta -= lr * theta.grad
                if bias.grad is not None:
                    bias -= lr * bias.grad
            
            theta.grad.zero_()
            if bias.grad is not None:
                bias.grad.zero_()

    y_pred = (x_q @ theta + bias).item()
    return y_pred, theta.detach(), bias.item()

# %%
# -------------------------
# LWLR in forma chiusa
# -------------------------
def lwlr_CF(x_q, X, y, t=0.5):
    m, d = X.shape

    # Aggiungi colonna di 1
    X_aug = torch.cat([X, torch.ones(m, 1)], dim=1)
    x_q_aug = torch.cat([x_q, torch.ones(1, 1)], dim=1)

    diff = X - x_q
    wi   = torch.exp(- (diff.norm(dim=1)**2) / (2 * t ** 2))
    W = torch.diag(wi)

    theta = torch.inverse(X_aug.T @ W @ X_aug) @ X_aug.T @ W @ y
    y_pred = (x_q_aug @ theta).item()
    return y_pred, theta[:-1], theta[-1]

# %%
# -------------------------
# Dati
# -------------------------
data_points = 50
X = torch.linspace(-5, 5, data_points).unsqueeze(1)  # (50,1)
y_true = 3.5 * X + 1.5

# Rumore sinusoidale
amplitude = 2.0
frequency = 1.0
y = y_true + amplitude * torch.sin(frequency * X)

# -------------------------
# Linear regression "globale"
# -------------------------
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x @ self.weights + self.bias

model = LinearModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(200):
    y_pred = model(X)
    loss_value = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

# %%
from tqdm import tqdm

# -------------------------
# Query points
# -------------------------
X_query = torch.linspace(-5, 5, 200).unsqueeze(1)

# Predizioni con SGD
y_preds_sgd = []
for x_q in tqdm(X_query, desc="LWLR - SGD"):
    y_pred, _, _ = lwlr_SGD(x_q.unsqueeze(0), X, y, t=1.0, lr=0.05, epochs=30)
    y_preds_sgd.append(y_pred)
y_preds_sgd = torch.tensor(y_preds_sgd)

# Predizioni con CF
y_preds_cf = []
for x_q in tqdm(X_query, desc="LWLR - Closed Form"):
    y_pred, _, _ = lwlr_CF(x_q.unsqueeze(0), X, y, t=1.0)
    y_preds_cf.append(y_pred)
y_preds_cf = torch.tensor(y_preds_cf)

# %%
# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(10,7))
plt.scatter(X.squeeze(), y.squeeze(), label="Dati con rumore sinusoidale", alpha=0.6)
plt.plot(torch.sort(X).values, (3.5 * torch.sort(X).values + 1.5), color="red", label="Funzione reale")
plt.plot(torch.sort(X).values, (model.weights.item() * torch.sort(X).values + model.bias.item()), 
         color="orange", label="Linear Regression (globale)")
plt.plot(X_query.squeeze(), y_preds_sgd, color="green", label="LWLR - SGD")
plt.plot(X_query.squeeze(), y_preds_cf, color="blue", linestyle="--", label="LWLR - Closed Form")
plt.xlabel("X")
plt.ylabel("y")
# plt.xlim(-5, -2)
# plt.ylim(-15, -5)
plt.legend()
plt.title("Confronto: regressione lineare vs locale (SGD e closed form)")
plt.show()

# %%
import torch.nn.functional as F
from tqdm import tqdm

# -------------------------
# Test set (uguale per tutti)
# -------------------------
X_test = torch.linspace(-5, 5, 200).unsqueeze(1)
y_true_test = 3.5 * X_test + 1.5
y_test = y_true_test + amplitude * torch.sin(frequency * X_test)

# 1) Funzione reale
mse_true = F.mse_loss(y_true_test, y_test).item()

# 2) Linear Regression (globale)
y_pred_linreg = model(X_test)
mse_linreg = F.mse_loss(y_pred_linreg, y_test).item()

# 3) LWLR con SGD
y_preds_sgd = []
for x_q in tqdm(X_test, desc="Calcolo MSE - LWLR SGD"):
    y_pred, _, _ = lwlr_SGD(x_q.unsqueeze(0), X, y, t=1.0, lr=0.05, epochs=30)
    y_preds_sgd.append(y_pred)
y_preds_sgd = torch.tensor(y_preds_sgd)
mse_sgd = F.mse_loss(y_preds_sgd, y_test).item()

# 4) LWLR con CF
y_preds_cf = []
for x_q in tqdm(X_test, desc="Calcolo MSE - LWLR Closed Form"):
    y_pred, _, _ = lwlr_CF(x_q.unsqueeze(0), X, y, t=1.0)
    y_preds_cf.append(y_pred)
y_preds_cf = torch.tensor(y_preds_cf)
mse_cf = F.mse_loss(y_preds_cf, y_test).item()

# -------------------------
# Risultati
# -------------------------
print(f"MSE funzione reale vs dati con rumore: {mse_true:.4f}")
print(f"MSE regressione lineare globale:      {mse_linreg:.4f}")
print(f"MSE LWLR con SGD:                     {mse_sgd:.4f}")
print(f"MSE LWLR con Closed Form:             {mse_cf:.4f}")

# %%



