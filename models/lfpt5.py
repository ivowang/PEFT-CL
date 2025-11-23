"""
LFPT5: Lifelong Few-shot Prompt Tuning
Based on the reference implementation but adapted for image classification
"""
import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import PromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
        # Create network with LFPT5 backbone
        self._network = PromptVitNet(args, True)
        
        # LFPT5 specific parameters
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.args = args
        
        # Training epochs
        self.tuned_epoch = args.get("tuned_epoch", 5)
        
        # Knowledge distillation parameters
        self.kd_lambda = args.get("lfpt5_kd_lambda", 0.5)
        self.use_kd = args.get("lfpt5_use_kd", True)
        
        # Memory replay (optional)
        self.memory_size = args.get("memory_size", 0)
        self.use_memory = self.memory_size > 0
        
        # Freeze the parameters for ViT
        if self.args.get("freeze", False):
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
            
            # freeze specified parameters
            freeze_list = self.args.get("freeze", [])
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(freeze_list)):
                    p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')
        
        # Print trainable parameter names
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))
        
        # Store previous task's prompt for knowledge distillation
        self.previous_prompt = None
        
        # Initialize memory for replay
        self._memory_data = []
        self._memory_targets = []
    
    def after_task(self):
        self._known_classes = self._total_classes
        
        # Save current prompt for next task's knowledge distillation
        if hasattr(self._network.backbone, 'prompt'):
            self.previous_prompt = self._network.backbone.prompt.prompt_embedding.data.clone().detach()
            logging.info("Saved prompt for task {} for knowledge distillation".format(self._cur_task))
        
        # Update memory if using memory replay
        if self.use_memory and self._cur_task >= 0:
            self._update_memory()
    
    def _update_memory(self):
        """Update memory with samples from current task"""
        # Get current task data
        current_data = []
        current_targets = []
        
        # Collect samples from current task
        if hasattr(self, 'train_dataset'):
            for idx in range(min(self.memory_size // (self._cur_task + 1), len(self.train_dataset))):
                _, img, target = self.train_dataset[idx]
                current_data.append(img.numpy())
                current_targets.append(target.item())
        
        # Add to memory
        if len(current_data) > 0:
            self._memory_data.extend(current_data)
            self._memory_targets.extend(current_targets)
            
            # Limit memory size
            if len(self._memory_data) > self.memory_size:
                # Keep oldest samples
                self._memory_data = self._memory_data[-self.memory_size:]
                self._memory_targets = self._memory_targets[-self.memory_size:]
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train"
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        
        # Add memory samples to training data if using memory
        if self.use_memory and len(self._memory_data) > 0:
            from utils.data import iData
            memory_dataset = iData(
                np.array(self._memory_data),
                np.array(self._memory_targets),
                transform=train_dataset.transform,
                class_map=train_dataset.class_map
            )
            from torch.utils.data import ConcatDataset
            train_dataset = ConcatDataset([train_dataset, memory_dataset])
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Set previous prompt for knowledge distillation
        if self._cur_task > 0 and hasattr(self._network.backbone, 'prompt') and self.previous_prompt is not None:
            self._network.backbone.prompt.set_previous_prompt(self.previous_prompt)
            logging.info("Set previous prompt for knowledge distillation")
        
        if len(self._multiple_gpus) > 1:
            logging.info('Using Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        
        self._init_train(train_loader, test_loader, optimizer, scheduler)
    
    def get_optimizer(self):
        """Get optimizer for training"""
        # Only optimize prompt parameters and classifier head
        params = []
        if hasattr(self._network.backbone, 'prompt') and hasattr(self._network.backbone.prompt, 'prompt_embedding'):
            params.append({'params': self._network.backbone.prompt.prompt_embedding, 'lr': self.init_lr})
        if hasattr(self._network.backbone, 'head'):
            params.append({'params': self._network.backbone.head.parameters(), 'lr': self.init_lr})
        
        if len(params) == 0:
            # Fallback: optimize all trainable parameters
            params = [{'params': filter(lambda p: p.requires_grad, self._network.backbone.parameters()), 'lr': self.init_lr}]
        
        if self.args.get('optimizer') == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args.get('optimizer') == 'adam':
            optimizer = optim.Adam(params, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args.get('optimizer') == 'adamw':
            optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        else:
            optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        
        return optimizer
    
    def get_scheduler(self, optimizer):
        """Get learning rate scheduler"""
        if self.args.get("scheduler") == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.tuned_epoch,
                eta_min=self.min_lr
            )
        elif self.args.get("scheduler") == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args.get("init_milestones", []),
                gamma=self.args.get("init_lr_decay", 0.1)
            )
        elif self.args.get("scheduler") == 'constant':
            scheduler = None
        else:
            scheduler = None
        
        return scheduler
    
    def _compute_kd_loss(self, current_logits, previous_logits, temperature=4.0):
        """Compute knowledge distillation loss using KL divergence"""
        if previous_logits is None:
            return torch.tensor(0.0, device=current_logits.device)
        
        # Apply temperature scaling
        current_probs = F.softmax(current_logits / temperature, dim=1)
        previous_probs = F.softmax(previous_logits / temperature, dim=1)
        
        # Add small epsilon to avoid log(0)
        current_probs = current_probs + 1e-8
        previous_probs = previous_probs + 1e-8
        
        # Compute KL divergence
        kd_loss = F.kl_div(
            F.log_softmax(current_logits / temperature, dim=1),
            previous_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return kd_loss
    
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """Training loop"""
        prog_bar = tqdm(range(self.tuned_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            if hasattr(self._network, 'original_backbone'):
                self._network.original_backbone.eval()
            
            losses = 0.0
            kd_losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # Forward pass with current prompt
                output = self._network(inputs, task_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                
                # Mask out old classes during training (optional)
                if self.args.get("mask_old_classes", False):
                    logits[:, :self._known_classes] = float('-inf')
                
                loss = F.cross_entropy(logits, targets.long())
                
                # Knowledge distillation loss
                kd_loss = torch.tensor(0.0, device=self._device)
                if self.use_kd and self._cur_task > 0 and self.previous_prompt is not None:
                    # Forward pass with previous prompt
                    with torch.no_grad():
                        output_kd = self._network(inputs, task_id=self._cur_task, train=False, use_kd=True)
                        if output_kd is not None and "logits" in output_kd:
                            logits_kd = output_kd["logits"][:, :self._total_classes]
                            # Only compute KD loss on new task samples (not memory samples)
                            # For simplicity, compute on all samples
                            kd_loss = self._compute_kd_loss(logits, logits_kd)
                
                # Combined loss
                finalloss = loss * (1.0 - self.kd_lambda) + kd_loss * self.kd_lambda
                
                optimizer.zero_grad()
                finalloss.backward()
                optimizer.step()
                
                losses += loss.item()
                kd_losses += kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss
                
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            if scheduler:
                scheduler.step()
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if (epoch + 1) % 5 == 0 or epoch == self.tuned_epoch - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, KD_Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.tuned_epoch,
                    losses / len(train_loader),
                    kd_losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, KD_Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.tuned_epoch,
                    losses / len(train_loader),
                    kd_losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        
        logging.info(info)
    
    def _eval_cnn(self, loader):
        """Evaluate on test set"""
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        
        return np.concatenate(y_pred), np.concatenate(y_true)
    
    def _compute_accuracy(self, model, loader):
        """Compute accuracy on a loader"""
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

