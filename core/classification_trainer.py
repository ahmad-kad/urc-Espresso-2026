#!/usr/bin/env python3
"""
Classification Trainer for torchvision models
Trains classification models on the consolidated dataset
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from core.data.classification_dataset import create_classification_data_loaders
from utils.logger_config import get_logger

logger = get_logger(__name__)


class ClassificationTrainer:
    """
    Trainer for torchvision classification models
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[Any] = None

        logger.info(f"Classification trainer initialized on device: {self.device}")

    def setup_model(self, model: nn.Module) -> None:
        """Setup the model for training"""
        self.model = model.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        lr = self.config.get("training", {}).get("learning_rate", 0.001)
        weight_decay = self.config.get("training", {}).get("weight_decay", 1e-4)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        logger.info(f"Model setup complete with learning rate: {lr}")

    def setup_data_loaders(self) -> None:
        """Setup data loaders for classification"""
        data_dir = self.config.get("data", {}).get("path", "consolidated_dataset")
        batch_size = self.config.get("training", {}).get("batch_size", 32)
        num_workers = self.config.get("training", {}).get("num_workers", 4)
        input_size = self.config.get("model", {}).get("input_size", 224)

        self.train_loader, self.val_loader, self.test_loader = (
            create_classification_data_loaders(
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                target_size=(input_size, input_size),
            )
        )

        logger.info(f"Data loaders created with batch size: {batch_size}")

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0

        # Progress bar for training batches
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Update progress bar
            batch_loss = loss.item()
            batch_acc = (torch.sum(preds == labels.data).item()) / inputs.size(0)
            progress_bar.set_postfix({"loss": ".4f", "acc": ".3f"})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.train_loader.dataset)

        return epoch_loss, epoch_acc.item()

    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        # Progress bar for validation batches
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                batch_loss = loss.item()
                batch_acc = (torch.sum(preds == labels.data).item()) / inputs.size(0)
                progress_bar.set_postfix({"loss": ".4f", "acc": ".3f"})

        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.val_loader.dataset)

        return epoch_loss, epoch_acc.item()

    def train(self, num_epochs: int = 25, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Train the model"""
        logger.info(f"Starting training for {num_epochs} epochs")

        best_model_wts = self.model.state_dict()
        best_acc = 0.0
        history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        # Progress bar for epochs
        epoch_progress = tqdm(range(num_epochs), desc="Epochs", unit="epoch")

        for epoch in epoch_progress:
            epoch_start = time.time()

            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = self.model.state_dict().copy()

            # Update progress bar with epoch results
            epoch_time = time.time() - epoch_start
            epoch_progress.set_postfix(
                {
                    "train_loss": ".4f",
                    "train_acc": ".3f",
                    "val_loss": ".4f",
                    "val_acc": ".3f",
                    "best_acc": ".3f",
                    "time": ".1f",
                }
            )

            # Log to file (not console to avoid cluttering tqdm output)
            logger.info(".4f" ".4f")

            # Store history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

        # Save model if path provided
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Best model saved to: {save_path}")

        # Final test evaluation
        test_loss, test_acc = self.test()

        results = {
            "best_val_acc": best_acc,
            "final_test_acc": test_acc,
            "final_test_loss": test_loss,
            "history": history,
            "num_epochs": num_epochs,
        }

        logger.info(".4f" ".4f")

        return results

    def test(self) -> Tuple[float, float]:
        """Test the model on test set"""
        test_size = len(self.test_loader.dataset)

        if test_size == 0:
            logger.warning(
                "No test samples available, using validation set for testing"
            )
            # Use validation set if test set is empty
            self.model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Progress bar for test batches
            progress_bar = tqdm(self.val_loader, desc="Testing", leave=False)

            with torch.no_grad():
                for inputs, labels in progress_bar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # Update progress bar
                    batch_loss = loss.item()
                    batch_acc = (torch.sum(preds == labels.data).item()) / inputs.size(
                        0
                    )
                    progress_bar.set_postfix({"loss": ".4f", "acc": ".3f"})

            epoch_loss = running_loss / len(self.val_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.val_loader.dataset)
        else:
            self.model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Progress bar for test batches
            progress_bar = tqdm(self.test_loader, desc="Testing", leave=False)

            with torch.no_grad():
                for inputs, labels in progress_bar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # Update progress bar
                    batch_loss = loss.item()
                    batch_acc = (torch.sum(preds == labels.data).item()) / inputs.size(
                        0
                    )
                    progress_bar.set_postfix({"loss": ".4f", "acc": ".3f"})

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

        return epoch_loss, epoch_acc.item()

    def save_checkpoint(self, path: str, epoch: int, loss: float, acc: float) -> None:
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "loss": loss,
            "accuracy": acc,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        acc = checkpoint["accuracy"]

        logger.info(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}, acc {acc:.4f}")
        return epoch, loss, acc
