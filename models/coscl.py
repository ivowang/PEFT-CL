"""
CoSCL: Cooperation of Small Continual Learners
Based on the reference implementation but adapted for the framework
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy

from models.base import BaseLearner
from backbone.coscl_net import CoSCLNet
from utils.coscl_utils import KLD, fisher_matrix_diag_coscl
from utils.toolkit import tensor2numpy, accuracy


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
        # CoSCL specific parameters
        self.lamb = args.get("coscl_lamb", 10000.0)  # EWC regularization weight
        self.lamb1 = args.get("coscl_lamb1", 0.02)   # KLD loss weight
        self.s_gate = args.get("coscl_s_gate", 100.0)  # Task gate scaling
        self.use_TG = args.get("coscl_use_TG", False)  # Use task-adaptive gate
        
        # Training parameters
        self.nepochs = args.get("coscl_nepochs", 100)
        self.lr = args.get("coscl_lr", 0.001)
        self.lr_min = args.get("coscl_lr_min", 1e-6)
        self.lr_factor = args.get("coscl_lr_factor", 3)
        self.lr_patience = args.get("coscl_lr_patience", 5)
        self.batch_size = args.get("batch_size", 256)
        self.num_workers = args.get("num_workers", 4)
        
        # Initialize model components
        self.coscl_model = None
        self.fisher = None
        self.old_param = None
        self.kld = KLD()
        self.ce = nn.CrossEntropyLoss()
        
        self.test_loader = None

    def _initialize_model(self, data_manager):
        """Initialize the CoSCL network."""
        if self.coscl_model is not None:
            return
        
        # Get input size from dataset
        # For CIFAR-100, input size is (3, 32, 32)
        dataset_name = self.args.get("dataset", "").lower()
        if "cifar100" in dataset_name or "cifar" in dataset_name:
            inputsize = (3, 32, 32)
        else:
            # Default to CIFAR size
            inputsize = (3, 32, 32)
            logging.warning(f"Unknown dataset {dataset_name}, using default input size {inputsize}")
        
        # Create task-class mapping
        taskcla = []
        for task_idx in range(data_manager.nb_tasks):
            task_size = data_manager.get_task_size(task_idx)
            taskcla.append((task_idx, task_size))
        
        # Create CoSCL network
        self.coscl_model = CoSCLNet(inputsize, taskcla, self.use_TG)
        self.coscl_model.s_gate = self.s_gate
        self.coscl_model.to(self._device)
        
        # Set as network for framework compatibility
        self._network = self.coscl_model
        
        logging.info(f"Initialized CoSCL model with {len(taskcla)} tasks")

    def _get_optimizer(self, lr=None):
        """Get optimizer for training."""
        if lr is None:
            lr = self.lr
        optimizer = torch.optim.Adam(self.coscl_model.parameters(), lr=lr)
        return optimizer

    def incremental_train(self, data_manager):
        """Train on a new task."""
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        # Initialize model if needed
        self._initialize_model(data_manager)
        
        # Update output layers for new classes
        self._update_output_layers(data_manager)
        
        # Get training data
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        # Get validation data
        val_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        # Get test data
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        # Train the model
        self._train_task(train_loader, val_loader)
        
        # Update Fisher information matrix
        self._update_fisher(train_loader)

    def _update_output_layers(self, data_manager):
        """Update output layers for new classes."""
        # Get current task size
        task_size = data_manager.get_task_size(self._cur_task)
        
        # Check if we need to add new output layer
        if self._cur_task >= len(self.coscl_model.last):
            # Add new output layer for this task
            new_layer = nn.Linear(256, task_size).to(self._device)
            self.coscl_model.last.append(new_layer)
        else:
            # Update existing layer if needed
            existing_size = self.coscl_model.last[self._cur_task].out_features
            if existing_size != task_size:
                new_layer = nn.Linear(256, task_size).to(self._device)
                self.coscl_model.last[self._cur_task] = new_layer

    def _train_task(self, train_loader, val_loader):
        """Train on current task."""
        best_loss = np.inf
        best_model = deepcopy(self.coscl_model.state_dict())
        lr = self.lr
        optimizer = self._get_optimizer(lr)
        patience = self.lr_patience
        
        for epoch in range(self.nepochs):
            # Training
            train_loss, train_acc = self._train_epoch(train_loader, optimizer)
            
            # Validation
            val_loss, val_acc = self._eval_task_internal(val_loader)
            
            logging.info(
                f"Epoch {epoch+1}/{self.nepochs} | "
                f"Train: loss={train_loss:.3f}, acc={train_acc*100:.1f}% | "
                f"Val: loss={val_loss:.3f}, acc={val_acc*100:.1f}% | "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Learning rate scheduling
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.coscl_model.state_dict())
                patience = self.lr_patience
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    if lr < self.lr_min:
                        logging.info("Learning rate reached minimum")
                    else:
                        optimizer = self._get_optimizer(lr)
                        patience = self.lr_patience
        
        # Restore best model
        self.coscl_model.load_state_dict(best_model)
        
        # Task-adaptive gate update
        if self.use_TG:
            task = torch.LongTensor([self._cur_task]).to(self._device)
            mask = self.coscl_model.mask(task, s=self.coscl_model.s_gate)
            for i in range(len(mask)):
                mask[i] = torch.autograd.Variable(mask[i].data.clone(), requires_grad=False)
        
        # Save old parameters for EWC
        self.old_param = {}
        for n, p in self.coscl_model.named_parameters():
            self.old_param[n] = p.data.clone().detach()

    def _train_epoch(self, train_loader, optimizer):
        """Train for one epoch."""
        self.coscl_model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_num = 0
        
        for _, inputs, targets in train_loader:
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            task = torch.LongTensor([self._cur_task]).to(self._device)
            
            # Forward
            outputs, outputs_expert, _ = self.coscl_model.forward(
                inputs, task, return_expert=True
            )
            
            # Map global class indices to task-local indices for loss computation
            if self._cur_task == 0:
                task_local_targets = targets
            else:
                task_local_targets = targets - self._known_classes
            
            # Compute loss
            loss_ce = self.ce(outputs, task_local_targets)
            loss_kld = self.kld(outputs_expert)
            loss_reg = self._compute_ewc_loss()
            
            loss = loss_ce + self.lamb1 * loss_kld + self.lamb * loss_reg
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            _, pred = outputs.max(1)
            hits = (pred == task_local_targets).float()
            
            total_loss += loss.item() * len(targets)
            total_acc += hits.sum().item()
            total_num += len(targets)
        
        return total_loss / total_num, total_acc / total_num

    def _compute_ewc_loss(self):
        """Compute EWC regularization loss."""
        if self._cur_task == 0 or self.fisher is None or self.old_param is None:
            return torch.tensor(0.0, device=self._device)
        
        loss_reg = 0.0
        for name, param in self.coscl_model.named_parameters():
            if name in self.fisher and name in self.old_param:
                loss_reg += torch.sum(
                    self.fisher[name] * (self.old_param[name] - param).pow(2)
                ) / 2
        
        return loss_reg

    def _update_fisher(self, train_loader):
        """Update Fisher information matrix."""
        # Convert DataLoader to tensor format for Fisher computation
        all_images = []
        all_targets = []
        
        for _, images, targets in train_loader:
            all_images.append(images)
            all_targets.append(targets)
        
        if len(all_images) == 0:
            return
        
        xtrain = torch.cat(all_images, dim=0).to(self._device)
        ytrain_global = torch.cat(all_targets, dim=0).to(self._device)
        
        # Convert global class indices to task-local indices
        # CoSCL uses task-specific output layers, so we need task-local indices
        if self._cur_task == 0:
            ytrain = ytrain_global
        else:
            ytrain = ytrain_global - self._known_classes
        
        # Ensure targets are within valid range
        task_size = self.args["increment"] if self._cur_task > 0 else self.args["init_cls"]
        ytrain = torch.clamp(ytrain, 0, task_size - 1)
        
        # Compute Fisher matrix
        # Note: fisher_matrix_diag_coscl uses CrossEntropyLoss internally
        fisher_new = fisher_matrix_diag_coscl(
            self._cur_task,
            xtrain,
            ytrain,
            self.coscl_model,
            self.ce,
            sbatch=self.batch_size,
            device=self._device,
        )
        
        # Merge with previous Fisher matrices
        if self.fisher is not None and self._cur_task > 0:
            fisher_old = {}
            for n in self.coscl_model.named_parameters():
                if n[0] in self.fisher:
                    fisher_old[n[0]] = self.fisher[n[0]].clone()
            
            for n, _ in self.coscl_model.named_parameters():
                if n in fisher_old:
                    fisher_new[n] = fisher_new[n] + fisher_old[n]
        
        self.fisher = fisher_new

    def _eval_task_internal(self, loader):
        """Internal evaluation function."""
        self.coscl_model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_num = 0
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                targets_tensor = targets.to(self._device)
                task = torch.LongTensor([self._cur_task]).to(self._device)
                
                outputs = self.coscl_model.forward(inputs, task)
                
                # Map global class indices to task-local indices for loss computation
                # For current task, classes start from self._known_classes
                if self._cur_task == 0:
                    task_local_targets = targets_tensor
                else:
                    task_local_targets = targets_tensor - self._known_classes
                
                loss = self.ce(outputs, task_local_targets)
                
                _, pred = outputs.max(1)
                hits = (pred == task_local_targets).float()
                
                total_loss += loss.item() * len(targets)
                total_acc += hits.sum().item()
                total_num += len(targets)
        
        return total_loss / total_num, total_acc / total_num

    def eval_task(self):
        """Evaluate on all seen tasks."""
        if self.test_loader is None:
            return None, None
        
        self.coscl_model.eval()
        y_pred, y_true = [], []
        
        # Build class-to-task mapping
        class_to_task = {}
        class_offset = 0
        for task_idx in range(self._cur_task + 1):
            task_size = self.args["increment"] if task_idx > 0 else self.args["init_cls"]
            for class_idx in range(class_offset, class_offset + task_size):
                class_to_task[class_idx] = task_idx
            class_offset += task_size
        
        with torch.no_grad():
            for _, inputs, targets in self.test_loader:
                inputs = inputs.to(self._device)
                targets_np = targets.cpu().numpy()
                
                # CoSCL uses task-specific output layers
                # For evaluation, we need to aggregate outputs from all tasks
                # and map task-local class indices to global class indices
                batch_size = inputs.size(0)
                num_classes = self._total_classes
                
                # Initialize combined logits with -inf
                combined_logits = torch.full(
                    (batch_size, num_classes), 
                    float('-inf'), 
                    device=self._device
                )
                
                # Get outputs from each task and map to global class indices
                for task_idx in range(self._cur_task + 1):
                    task = torch.LongTensor([task_idx]).to(self._device)
                    task_outputs = self.coscl_model.forward(inputs, task)
                    
                    # Map task-local classes to global classes
                    if task_idx == 0:
                        global_start = 0
                        global_end = self.args["init_cls"]
                    else:
                        global_start = self.args["init_cls"] + (task_idx - 1) * self.args["increment"]
                        global_end = global_start + self.args["increment"]
                    
                    # Copy task outputs to corresponding global positions
                    combined_logits[:, global_start:global_end] = task_outputs
                
                # Get top-k predictions
                predicts = torch.topk(
                    combined_logits, k=self.topk, dim=1, largest=True, sorted=True
                )[1]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets_np)
        
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        
        cnn_accy = self._evaluate(y_pred, y_true)
        
        # CoSCL doesn't use NME
        return cnn_accy, None

    def after_task(self):
        """Called after each task."""
        self._known_classes = self._total_classes

