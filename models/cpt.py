"""
CPT: Continual Prompt Tuning
Based on the reference implementation but adapted for the framework
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
        
        # Create network with CPT backbone
        self._network = PromptVitNet(args, True)
        
        # CPT specific parameters
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.first_lr = args.get("cpt_first_lr", args.get("init_lr", 0.001))
        self.meta_lr = args.get("cpt_meta_lr", args.get("init_lr", 0.001))
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.args = args
        
        # Training epochs
        self.tuned_epoch = args.get("tuned_epoch", 5)
        self.first_epochs = args.get("cpt_first_epochs", 20)
        self.meta_epochs = args.get("cpt_meta_epochs", 10)
        
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
        
        # Initialize memory for replay
        self._memory_data = []
        self._memory_targets = []
    
    def after_task(self):
        self._known_classes = self._total_classes
        
        # Update memory if using memory replay
        if self.use_memory and self._cur_task >= 0:
            self._update_memory()
    
    def _update_memory(self):
        """Update memory with samples from current task"""
        # Get current task data
        current_data = []
        current_targets = []
        
        # Collect samples from current task
        # This is a simplified version - in practice, you might want to use
        # a more sophisticated memory selection strategy
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
        
        if len(self._multiple_gpus) > 1:
            logging.info('Using Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        # Determine training mode based on task
        if self._cur_task == 0:
            # First task: train first_prompt
            self._train_first_prompt(train_loader, test_loader)
        else:
            # Subsequent tasks: train prompt, then meta_prompt
            self._train_prompt(train_loader, test_loader)
            self._train_meta_prompt(train_loader, test_loader)
    
    def _train_first_prompt(self, train_loader, test_loader):
        """Train first_prompt for task 0"""
        logging.info("Training first_prompt for task 0")
        
        # Set prompt to first_prompt mode
        if hasattr(self._network.backbone, 'prompt'):
            self._network.backbone.prompt.set_training_mode('first_prompt')
            self._network.backbone.prompt.set_current_task(0)
        
        optimizer = self.get_optimizer(lr=self.first_lr, mode='first_prompt')
        scheduler = self.get_scheduler(optimizer)
        
        self._init_train(train_loader, test_loader, optimizer, scheduler, epochs=self.first_epochs)
    
    def _train_prompt(self, train_loader, test_loader):
        """Train prompt for current task"""
        logging.info("Training prompt for task {}".format(self._cur_task))
        
        # Set prompt to prompt mode
        if hasattr(self._network.backbone, 'prompt'):
            self._network.backbone.prompt.set_training_mode('prompt')
            self._network.backbone.prompt.set_current_task(self._cur_task)
            
            # Initialize prompt from meta_prompt if available
            if self._cur_task > 0:
                self._network.backbone.prompt.initialize_prompt_from_meta(self._cur_task)
        
        optimizer = self.get_optimizer(lr=self.init_lr, mode='prompt')
        scheduler = self.get_scheduler(optimizer)
        
        self._init_train(train_loader, test_loader, optimizer, scheduler, epochs=self.tuned_epoch)
    
    def _train_meta_prompt(self, train_loader, test_loader):
        """Train meta_prompt after training prompt"""
        if self._cur_task >= self.args.get("nb_tasks", 10) - 1:
            # Last task, no need to train meta_prompt
            return
        
        logging.info("Training meta_prompt for task {}".format(self._cur_task))
        
        # Set prompt to meta_prompt mode
        if hasattr(self._network.backbone, 'prompt'):
            self._network.backbone.prompt.set_training_mode('meta_prompt')
            
            # Initialize meta_prompt from current task's prompt
            self._network.backbone.prompt.initialize_meta_prompt_from_prompt(self._cur_task)
        
        optimizer = self.get_optimizer(lr=self.meta_lr, mode='meta_prompt')
        scheduler = self.get_scheduler(optimizer)
        
        self._init_train(train_loader, test_loader, optimizer, scheduler, epochs=self.meta_epochs)
    
    def get_optimizer(self, lr=None, mode='prompt'):
        """Get optimizer for training"""
        if lr is None:
            lr = self.init_lr
        
        # Get parameters based on mode
        if mode == 'first_prompt':
            params = []
            if hasattr(self._network.backbone, 'prompt'):
                params.append({'params': self._network.backbone.prompt.first_prompt, 'lr': lr})
                if hasattr(self._network.backbone.prompt, 'prompt_key'):
                    params.append({'params': self._network.backbone.prompt.prompt_key, 'lr': lr})
        elif mode == 'meta_prompt':
            params = []
            if hasattr(self._network.backbone, 'prompt'):
                params.append({'params': self._network.backbone.prompt.meta_prompt, 'lr': lr})
        else:  # prompt mode
            params = []
            if hasattr(self._network.backbone, 'prompt'):
                # Get prompt parameters for current task
                if self._cur_task < self._network.backbone.prompt.pool_size:
                    # Use task-specific prompt
                    task_prompt = self._network.backbone.prompt.prompt[self._cur_task]
                    params.append({'params': task_prompt, 'lr': lr})
                else:
                    # Use all prompts
                    params.append({'params': self._network.backbone.prompt.prompt, 'lr': lr})
                if hasattr(self._network.backbone.prompt, 'prompt_key'):
                    params.append({'params': self._network.backbone.prompt.prompt_key, 'lr': lr})
        
        # Add classifier head parameters
        params.append({'params': self._network.backbone.head.parameters(), 'lr': lr})
        
        if self.args.get('optimizer') == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=lr, weight_decay=self.weight_decay)
        elif self.args.get('optimizer') == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        elif self.args.get('optimizer') == 'adamw':
            optimizer = optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        else:
            optimizer = optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        
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
    
    def _init_train(self, train_loader, test_loader, optimizer, scheduler, epochs=None):
        """Training loop"""
        if epochs is None:
            epochs = self.tuned_epoch
        
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            if hasattr(self._network, 'original_backbone'):
                self._network.original_backbone.eval()
            
            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                output = self._network(inputs, task_id=self._cur_task, train=True)
                
                # Handle different output formats
                if isinstance(output, dict):
                    # If output is a dictionary, get logits from it
                    if "logits" not in output:
                        raise KeyError(f"'logits' not found in output. Available keys: {output.keys()}")
                    logits = output["logits"]
                elif isinstance(output, torch.Tensor):
                    # If output is a tensor directly, use it as logits
                    logits = output
                else:
                    raise TypeError(f"Unexpected output type: {type(output)}. Expected dict or Tensor.")
                
                # Ensure logits is a tensor and has correct shape
                if not isinstance(logits, torch.Tensor):
                    raise TypeError(f"logits is not a tensor, got {type(logits)}")
                
                # Ensure logits is 2D: (batch_size, num_classes)
                if logits.dim() == 1:
                    # If logits is 1D, it might be a single sample, unsqueeze to add batch dimension
                    logits = logits.unsqueeze(0)
                elif logits.dim() == 0:
                    # If logits is 0D (scalar), this is unexpected
                    raise ValueError(f"logits is a scalar (0D tensor), which is unexpected. Shape: {logits.shape}")
                elif logits.dim() > 2:
                    # If logits has more than 2 dimensions, reshape it
                    logits = logits.view(logits.size(0), -1)
                
                # Slice to get only the classes we care about
                if logits.size(1) >= self._total_classes:
                    logits = logits[:, :self._total_classes]
                # If logits has fewer classes than _total_classes, use all available classes
                
                # Mask out old classes during training (optional, for better performance)
                if self.args.get("mask_old_classes", False):
                    logits[:, :self._known_classes] = float('-inf')
                
                loss = F.cross_entropy(logits, targets.long())
                
                # Add pull constraint loss if specified
                if self.args.get("pull_constraint", False) and isinstance(output, dict) and 'reduce_sim' in output:
                    loss = loss - self.args.get("pull_constraint_coeff", 0.1) * output['reduce_sim']
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            if scheduler:
                scheduler.step()
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
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

