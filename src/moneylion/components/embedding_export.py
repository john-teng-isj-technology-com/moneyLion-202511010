from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle

from src.moneylion import logger
from src.moneylion.entity.config_entity import DataEmbeddingConfig
from src.moneylion.utils.common import create_directories


class _EmbedNet(nn.Module):
    def __init__(
        self,
        cardinalities: List[int],
        numeric_dim: int,
        dim_rule: str = "sqrt",
    ) -> None:
        super().__init__()

        self.emb_layers = nn.ModuleList()
        emb_dim_total   = 0
        for card in cardinalities:
            if dim_rule.isdigit():
                emb_dim = int(dim_rule)
            else:                          # rule == "sqrt"
                emb_dim = int(math.sqrt(card))
            emb_dim_total += emb_dim
            self.emb_layers.append(nn.Embedding(card, emb_dim))

        self.numeric_bn = nn.BatchNorm1d(numeric_dim) if numeric_dim else nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(emb_dim_total + numeric_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, cat_idx: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        emb_vecs = [emb(cat_idx[:, i]) for i, emb in enumerate(self.emb_layers)]
        concat   = torch.cat(emb_vecs + [num], dim=1)
        return self.head(concat), torch.cat(emb_vecs, dim=1)  # (logit, embed_vec)


class EmbeddingExporter:
    def __init__(self, cfg: DataEmbeddingConfig) -> None:
        self.cfg = cfg
        create_directories([self.cfg.root_dir])

        pp_dir = self.cfg.preproc_dir
        self.X_train_cat = np.load(pp_dir / "train_cat.npy")
        self.X_train_num = np.load(pp_dir / "train_num.npy")
        self.y_train     = np.load(pp_dir / "train_y.npy")

        self.X_val_cat   = np.load(pp_dir / "val_cat.npy")
        self.X_val_num   = np.load(pp_dir / "val_num.npy")
        self.y_val       = np.load(pp_dir / "val_y.npy")

        # vocabulary sizes
        with open(pp_dir / "vocabs.json") as f:
            self.vocabs: Dict[str, Dict[str, int]] = json.load(f)
        self.cardinalities = [len(stoi) for stoi in self.vocabs.values()]
        self.numeric_dim   = self.X_train_num.shape[1]

        # ---------- model ----------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = _EmbedNet(
            self.cardinalities,
            self.numeric_dim,
            self.cfg.embedding_dim_rule
        ).to(self.device)

    # -------------------------------------------------------------- #
    def _dl(self, X_cat, X_num, y, shuffle_flag=True) -> DataLoader:
        Xc = torch.tensor(X_cat, dtype=torch.long)
        Xn = torch.tensor(X_num, dtype=torch.float32)
        yt = torch.tensor(y,     dtype=torch.float32).unsqueeze(1)
        ds = TensorDataset(Xc, Xn, yt)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=False,
        )

    def train(self) -> None:
        torch.manual_seed(self.cfg.random_state)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        bce = nn.BCEWithLogitsLoss()

        train_loader = self._dl(self.X_train_cat, self.X_train_num, self.y_train)
        val_loader   = self._dl(self.X_val_cat,   self.X_val_num,   self.y_val, False)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_loss = 0.0
            for Xc, Xn, y in train_loader:
                Xc, Xn, y = Xc.to(self.device), Xn.to(self.device), y.to(self.device)
                opt.zero_grad()
                logit, _ = self.model(Xc, Xn)
                loss = bce(logit, y)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(y)

            # simple val AUC / loss logging (optional)
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for Xc, Xn, y in val_loader:
                    Xc, Xn, y = Xc.to(self.device), Xn.to(self.device), y.to(self.device)
                    logit, _  = self.model(Xc, Xn)
                    val_loss += bce(logit, y).item() * len(y)
            logger.info(
                f"Epoch {epoch+1}/{self.cfg.epochs} "
                f"train_loss={epoch_loss/len(self.y_train):.4f} "
                f"val_loss={val_loss/len(self.y_val):.4f}"
            )

    def _export_row_embeddings(self, X_cat: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = DataLoader(
            torch.tensor(X_cat, dtype=torch.long),
            batch_size=4096,
            shuffle=False
        )
        all_vecs = []
        with torch.no_grad():
            for Xc in loader:
                Xc = Xc.to(self.device)
                _, emb = self.model(Xc, torch.zeros((len(Xc), self.numeric_dim), device=self.device))
                all_vecs.append(emb.cpu().numpy())
        return np.vstack(all_vecs)

    def export(self) -> None:
        # Save raw embedding matrices per column (optional, nice for inspection)
        embed_dir = self.cfg.root_dir / "embed_matrices"
        create_directories([embed_dir])
        for name, emb_layer in zip(self.vocabs.keys(), self.model.emb_layers):
            np.save(embed_dir / f"{name}.npy", emb_layer.weight.detach().cpu().numpy())

        # Build dense matrices = [emb_vec | numeric]
        for split in ["train", "val", "test"]:
            X_cat = np.load(self.cfg.preproc_dir / f"{split}_cat.npy")
            X_num = np.load(self.cfg.preproc_dir / f"{split}_num.npy")
            y     = np.load(self.cfg.preproc_dir / f"{split}_y.npy")

            emb_vec = self._export_row_embeddings(X_cat)
            X_dense = np.hstack([emb_vec, X_num]).astype("float32")

            np.save(self.cfg.root_dir / f"X_{split}.npy", X_dense)
            np.save(self.cfg.root_dir / f"y_{split}.npy", y.astype("int64"))
            logger.info("Exported %s dense matrix shape %s", split, X_dense.shape)

        # schema
        schema = {
            "embedded_dim_total": int(emb_vec.shape[1]),
            "numeric_dim": int(self.numeric_dim),
            "feature_order": [f"emb_{i}" for i in range(emb_vec.shape[1])]
                            + [f"num_{i}" for i in range(self.numeric_dim)]
        }
        with open(self.cfg.root_dir / "embed_schema.json", "w") as f:
            json.dump(schema, f, indent=2)

        logger.info("Embedding export complete. Artifacts saved under %s", self.cfg.root_dir)

    def run(self):
        self.train()
        self.export()
