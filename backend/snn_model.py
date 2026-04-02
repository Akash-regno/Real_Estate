"""
Spiking Neural Network (SNN) for Property Price Prediction
- Uses encoded features from the autoencoder
- Learns complex non-linear patterns via spike-based computation
- Implemented using snntorch library
"""
import numpy as np
import torch
import torch.nn as nn
import os

try:
    import snntorch as snn
    from snntorch import surrogate
    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False
    print("[SNN] snntorch not available, using fallback MLP model")

from config import SNN_CONFIG, MODEL_DIR


class SNNPricePredictor(nn.Module):
    """SNN-based price prediction model using Leaky Integrate-and-Fire neurons"""

    def __init__(self, input_size, hidden_size=None, num_steps=None, beta=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size or SNN_CONFIG["hidden_size"]
        self.num_steps = num_steps or SNN_CONFIG["num_steps"]
        self.beta = beta or SNN_CONFIG["beta"]

        if HAS_SNNTORCH:
            spike_grad = surrogate.fast_sigmoid(slope=25)

            # Layer 1: Input -> Hidden
            self.fc1 = nn.Linear(input_size, self.hidden_size)
            self.lif1 = snn.Leaky(beta=self.beta, spike_grad=spike_grad)

            # Layer 2: Hidden -> Hidden
            self.fc2 = nn.Linear(self.hidden_size, self.hidden_size // 2)
            self.lif2 = snn.Leaky(beta=self.beta, spike_grad=spike_grad)

            # Layer 3: Hidden -> Output (regression: 1 output)
            self.fc3 = nn.Linear(self.hidden_size // 2, 1)
            self.lif3 = snn.Leaky(beta=self.beta, spike_grad=spike_grad)
        else:
            # Fallback MLP
            self.net = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_size),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_size // 2),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size // 2, 1),
                nn.Sigmoid()
            )

        print(f"[SNN] Model initialized: input={input_size}, hidden={self.hidden_size}, steps={self.num_steps}")

    def forward(self, x):
        if HAS_SNNTORCH:
            # Initialize hidden states
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()

            # Record final membrane potential for regression
            mem_rec = []

            for step in range(self.num_steps):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)

                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)

                cur3 = self.fc3(spk2)
                spk3, mem3 = self.lif3(cur3, mem3)

                mem_rec.append(mem3)

            # Use average membrane potential as regression output
            mem_stack = torch.stack(mem_rec, dim=0)
            output = torch.mean(mem_stack, dim=0)
            return torch.sigmoid(output)
        else:
            return self.net(x)


class SNNTrainer:
    """Training and inference manager for SNN price predictor"""

    def __init__(self, input_size):
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SNNPricePredictor(input_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=SNN_CONFIG["learning_rate"]
        )
        self.criterion = nn.MSELoss()
        self.history = {"train_loss": [], "val_loss": []}

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None):
        """Train the SNN model"""
        epochs = epochs or SNN_CONFIG["epochs"]
        batch_size = batch_size or SNN_CONFIG["batch_size"]

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            self.history["train_loss"].append(avg_train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(X_val_t)
                    val_loss = self.criterion(val_output, y_val_t).item()
                self.history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"[SNN] Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 25 == 0:
                val_str = f", val_loss: {val_loss:.6f}" if X_val is not None else ""
                print(f"[SNN] Epoch {epoch + 1}/{epochs} - loss: {avg_train_loss:.6f}{val_str}")

        print(f"[SNN] Training complete. Best val loss: {best_val_loss:.6f}")
        return self.history

    def predict(self, X):
        """Predict prices"""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_t).cpu().numpy().flatten()
        return predictions

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))

        # R² score
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2)
        }
        print(f"[SNN] Evaluation - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
        return metrics

    def save_model(self):
        """Save model state"""
        torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, "snn_model.pth"))

    def load_model(self):
        """Load model state"""
        path = os.path.join(MODEL_DIR, "snn_model.pth")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print("[SNN] Model loaded")

    def get_model_info(self):
        """Return model architecture info"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
            "uses_snntorch": HAS_SNNTORCH,
            "hidden_size": self.model.hidden_size,
            "num_steps": self.model.num_steps,
        }
