"""PyTorch LSTM regressors for stock closing price forecasting.

Low-compute defaults, early stopping, and optional quantile heads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import math

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

# Fail fast with a clear message if torch is not available, so optional imports can catch it
if not TORCH_AVAILABLE:  # pragma: no cover
    raise ImportError("PyTorch is required for LSTM model support. Install with 'pip install torch'.")


@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 128
    use_quantiles: bool = False
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"


class LSTMRegressor(nn.Module):
    """LSTM with a point head and optional quantile heads (P10/P90)."""
    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.point_head = nn.Linear(cfg.hidden_size, 1)
        if cfg.use_quantiles:
            self.q10_head = nn.Linear(cfg.hidden_size, 1)
            self.q90_head = nn.Linear(cfg.hidden_size, 1)
        else:
            self.q10_head = None
            self.q90_head = None

    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        y = self.point_head(h_last)
        if self.q10_head is not None:
            y10 = self.q10_head(h_last)
            y90 = self.q90_head(h_last)
            return y.squeeze(-1), y10.squeeze(-1), y90.squeeze(-1)
        return y.squeeze(-1)


def quantile_loss(y_true, y_pred, q: float):
    err = y_true - y_pred
    return torch.max((q - 0.5) * err, (q - 1) * err).mean()


class LSTMTrainer:
    """Encapsulates training/inference utilities for LSTMRegressor."""
    def __init__(self, cfg: LSTMConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for --model lstm. Please install torch.")
        self.cfg = cfg
        self.model = LSTMRegressor(cfg).to(cfg.device)
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def _make_loader(self, X: torch.Tensor, y: torch.Tensor, shuffle: bool) -> torch.utils.data.DataLoader:
        ds = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
    ) -> None:
        device = self.cfg.device
        model = self.model
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=2, factor=0.5
        )

        loss_mse = nn.MSELoss()
        patience = 5
        best_val = math.inf
        patience_left = patience

        train_loader = self._make_loader(X_train.to(device), y_train.to(device), shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_loader = self._make_loader(X_val.to(device), y_val.to(device), shuffle=False)

        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
        amp_enabled = (device == "cuda")

        for epoch in range(self.cfg.epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    out = model(xb)
                    if isinstance(out, tuple):
                        yp, y10, y90 = out
                        loss = loss_mse(yp, yb)
                        loss += quantile_loss(yb, y10, 0.1) + quantile_loss(yb, y90, 0.9)
                    else:
                        loss = loss_mse(out, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                epoch_loss += loss.item() * len(xb)

            epoch_loss /= max(1, len(train_loader.dataset))

            # Validation
            val_loss = epoch_loss
            if val_loader is not None:
                model.eval()
                total, n = 0.0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        out = model(xb)
                        if isinstance(out, tuple):
                            yp, y10, y90 = out
                            l = loss_mse(yp, yb)
                            l += quantile_loss(yb, y10, 0.1) + quantile_loss(yb, y90, 0.9)
                        else:
                            l = loss_mse(out, yb)
                        total += l.item() * len(xb)
                        n += len(xb)
                val_loss = total / max(1, n)
                lr_sched.step(val_loss)

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    patience_left = patience
                    self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        # Load best
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        device = self.cfg.device
        self.model.eval()
        with torch.no_grad():
            out = self.model(X.to(device))
        if isinstance(out, tuple):
            yp, y10, y90 = out
            return yp.cpu(), y10.cpu(), y90.cpu()
        return out.cpu(), None, None

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.cfg.device))
        self.model.to(self.cfg.device)


