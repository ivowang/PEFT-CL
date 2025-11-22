import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import HiDePromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

num_workers = 8

# Global variables for orth_loss (similar to HiDe-Prompt's implementation)
cls_mean = dict()
cls_cov = dict()


def orth_loss(features, targets, device, args):
    """Contrastive regularization loss for HiDe-Prompt"""
    global cls_mean
    if cls_mean:
        # orth loss of this batch
        sample_mean = []
        for k, v in cls_mean.items():
            if isinstance(v, list):
                sample_mean.extend(v)
            else:
                sample_mean.append(v)
        sample_mean = torch.stack(sample_mean, dim=0).to(device, non_blocking=True)
        M = torch.cat([sample_mean, features], dim=0)
        sim = torch.matmul(M, M.t()) / 0.8
        loss = F.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(device))
        return args.get("reg", 0.01) * loss
    else:
        sim = torch.matmul(features, features.t()) / 0.8
        loss = F.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(device))
        return args.get("reg", 0.01) * loss


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self._network = HiDePromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

        # Freeze the parameters for ViT.
        if self.args.get("freeze"):
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
        
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args.get("freeze", []))):
                    p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
            
        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args.get("reinit_optimizer", True):
            optimizer = self.get_optimizer()
            
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self):
        # Use larger prompt lr if specified
        if self.args.get("larger_prompt_lr", False):
            base_params = [p for name, p in self._network.backbone.named_parameters() if 'prompt' in name and p.requires_grad == True]
            base_fc_params = [p for name, p in self._network.backbone.named_parameters() if 'prompt' not in name and p.requires_grad == True]
            base_params = {'params': base_params, 'lr': self.init_lr, 'weight_decay': self.weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': self.init_lr * 0.1, 'weight_decay': self.weight_decay}
            network_params = [base_params, base_fc_params]
        else:
            network_params = filter(lambda p: p.requires_grad, self._network.parameters())
            
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                network_params, 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                network_params,
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                network_params,
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr' or self.args["scheduler"] == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args.get("init_milestones", [10]), gamma=self.args.get("init_lr_decay", 0.1))
        elif self.args["scheduler"] == 'constant':
            scheduler = None
        else:
            scheduler = None

        return scheduler

    def _init_prompt(self, optimizer):
        args = self.args
        model = self._network.backbone
        task_id = self._cur_task

        # Transfer previous learned prompt params to the new prompt
        if args.get("prompt_pool") and args.get("shared_prompt_pool", True):
            prev_start = (task_id - 1) * args.get("top_k", 1)
            prev_end = task_id * args.get("top_k", 1)

            cur_start = prev_end
            cur_end = (task_id + 1) * args.get("top_k", 1)

            if (prev_end > args.get("size", 10)) or (cur_end > args.get("size", 10)):
                pass
            else:
                use_prefix = args.get("use_prefix_tune_for_e_prompt", True)
                cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if use_prefix else (slice(None), slice(cur_start, cur_end))
                prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if use_prefix else (slice(None), slice(prev_start, prev_end))

                with torch.no_grad():
                    model.e_prompt.prompt.grad.zero_()
                    model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                
        # Transfer previous learned prompt param keys to the new prompt
        if args.get("prompt_pool") and args.get("shared_prompt_key", False):
            prev_start = (task_id - 1) * args.get("top_k", 1)
            prev_end = task_id * args.get("top_k", 1)

            cur_start = prev_end
            cur_end = (task_id + 1) * args.get("top_k", 1)

            if (prev_end > args.get("size", 10)) or (cur_end > args.get("size", 10)):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

                with torch.no_grad():
                    model.e_prompt.prompt_key.grad.zero_()
                    model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]

    def _compute_prompt_id(self, inputs, targets):
        """Compute prompt_id from original_model for HiDe-Prompt"""
        with torch.no_grad():
            original_output = self._network.original_backbone(inputs)
            logits = original_output['logits']
            
            # Mask out classes from non-current tasks
            if self.args.get("train_mask", True):
                mask = []
                for id in range(self._cur_task + 1):
                    task_classes = self.data_manager.get_task_size(id)
                    start = sum(self.data_manager.get_task_size(i) for i in range(id))
                    mask.extend(range(start, start + task_classes))
                not_mask = np.setdiff1d(np.arange(self._total_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(inputs.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            
            # Get predicted class and map to task_id
            pred_class = torch.max(logits, dim=1)[1]
            # Map class to task_id
            prompt_id = []
            for cls in pred_class:
                task_id = 0
                cumsum = 0
                for t in range(self._cur_task + 1):
                    task_size = self.data_manager.get_task_size(t)
                    if cls < cumsum + task_size:
                        task_id = t
                        break
                    cumsum += task_size
                prompt_id.append(task_id)
            prompt_id = torch.tensor(prompt_id, device=inputs.device).unsqueeze(-1)
            return prompt_id

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._network.original_backbone.eval()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                # Compute prompt_id from original_model
                prompt_id = self._compute_prompt_id(inputs, targets)
                
                output = self._network(
                    inputs, 
                    task_id=self._cur_task, 
                    prompt_id=prompt_id,
                    train=True,
                    prompt_momentum=self.args.get("prompt_momentum", 0.01)
                )
                logits = output["logits"][:, :self._total_classes]
                
                # Mask out classes from non-current task during training
                if self.args.get("train_mask", True):
                    logits[:, :self._known_classes] = float('-inf')

                loss = F.cross_entropy(logits, targets.long())
                
                # Add orth_loss (contrastive regularization)
                if 'pre_logits' in output:
                    loss += orth_loss(output['pre_logits'], targets, self._device, self.args)

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

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # Compute prompt_id for evaluation
                prompt_id = self._compute_prompt_id(inputs, targets)
                outputs = self._network(
                    inputs, 
                    task_id=self._cur_task,
                    prompt_id=prompt_id
                )["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # Compute prompt_id for evaluation
                prompt_id = self._compute_prompt_id(inputs, targets)
                outputs = model(
                    inputs, 
                    task_id=self._cur_task,
                    prompt_id=prompt_id
                )["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

