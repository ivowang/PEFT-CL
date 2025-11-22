"""
CoSCL Network: Cooperation of Small Continual Learners
Based on the reference implementation but adapted for the framework
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoSCLNet(nn.Module):
    """
    CoSCL Network with multiple expert networks.
    Each expert is a small CNN that processes the input independently.
    """
    def __init__(self, inputsize, taskcla, use_TG=False):
        super().__init__()
        ncha, size, _ = inputsize
        self.taskcla = taskcla
        self.nExpert = 5
        self.nc = 8
        self.last = torch.nn.ModuleList()
        self.s_gate = 1
        self.use_TG = use_TG

        # Expert 1
        self.net1 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc1 = nn.Linear(self.nc*64, 256)
        if self.use_TG:
            self.efc1 = torch.nn.Embedding(len(self.taskcla), 256)

        # Expert 2
        self.net2 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc2 = nn.Linear(self.nc*64, 256)
        if self.use_TG:
            self.efc2 = torch.nn.Embedding(len(self.taskcla), 256)

        # Expert 3
        self.net3 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc3 = nn.Linear(self.nc*64, 256)
        if self.use_TG:
            self.efc3 = torch.nn.Embedding(len(self.taskcla), 256)

        # Expert 4
        self.net4 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc4 = nn.Linear(self.nc*64, 256)
        if self.use_TG:
            self.efc4 = torch.nn.Embedding(len(self.taskcla), 256)

        # Expert 5
        self.net5 = nn.Sequential(
            nn.Conv2d(ncha, self.nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc, self.nc, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.nc, self.nc*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc*2, self.nc*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(self.nc*2, self.nc*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.nc*4, self.nc*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc5 = nn.Linear(self.nc*64, 256)
        if self.use_TG:
            self.efc5 = torch.nn.Embedding(len(self.taskcla), 256)

        # Task-specific output layers
        for t, n in self.taskcla:
            self.last.append(torch.nn.Linear(256, n))

        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.sig_gate = torch.nn.Sigmoid()
        
        # For feature extraction
        self.out_dim = 256

    def forward(self, x, t=None, return_expert=False, avg_act=False):
        """
        Forward pass through the CoSCL network.
        
        Args:
            x: Input tensor
            t: Task index (required if use_TG is True)
            return_expert: If True, return individual expert outputs
            avg_act: If True, save activations for gradient computation
        """
        if self.use_TG and t is not None:
            # With task-adaptive gate
            masks = self.mask(t, s=self.s_gate)
            gfc1, gfc2, gfc3, gfc4, gfc5 = masks

            Experts = []
            Experts_feature = []

            h1 = self.net1(x)
            h1 = h1.view(x.shape[0], -1)
            Experts_feature.append(h1)
            h1 = self.relu(self.fc1(h1))
            h1 = self.drop2(h1)
            h1 = h1 * gfc1.expand_as(h1)
            Experts.append(h1.unsqueeze(0))

            h2 = self.net2(x)
            h2 = h2.view(x.shape[0], -1)
            Experts_feature.append(h2)
            h2 = self.relu(self.fc2(h2))
            h2 = self.drop2(h2)
            h2 = h2 * gfc2.expand_as(h2)
            Experts.append(h2.unsqueeze(0))

            h3 = self.net3(x)
            h3 = h3.view(x.shape[0], -1)
            Experts_feature.append(h3)
            h3 = self.relu(self.fc3(h3))
            h3 = self.drop2(h3)
            h3 = h3 * gfc3.expand_as(h3)
            Experts.append(h3.unsqueeze(0))

            h4 = self.net4(x)
            h4 = h4.view(x.shape[0], -1)
            Experts_feature.append(h4)
            h4 = self.relu(self.fc4(h4))
            h4 = self.drop2(h4)
            h4 = h4 * gfc4.expand_as(h4)
            Experts.append(h4.unsqueeze(0))

            h5 = self.net5(x)
            h5 = h5.view(x.shape[0], -1)
            Experts_feature.append(h5)
            h5 = self.relu(self.fc5(h5))
            h5 = self.drop2(h5)
            h5 = h5 * gfc5.expand_as(h5)
            Experts.append(h5.unsqueeze(0))

            h = torch.cat([h_result for h_result in Experts], 0)
            h = torch.sum(h, dim=0).squeeze(0)

        else:
            # Without task-adaptive gate
            Experts = []
            Experts_feature = []

            h1 = self.net1(x)
            h1 = h1.view(x.shape[0], -1)
            Experts_feature.append(h1)
            h1 = self.relu(self.fc1(h1))
            h1 = self.drop2(h1)
            Experts.append(h1.unsqueeze(0))

            h2 = self.net2(x)
            h2 = h2.view(x.shape[0], -1)
            Experts_feature.append(h2)
            h2 = self.relu(self.fc2(h2))
            h2 = self.drop2(h2)
            Experts.append(h2.unsqueeze(0))

            h3 = self.net3(x)
            h3 = h3.view(x.shape[0], -1)
            Experts_feature.append(h3)
            h3 = self.relu(self.fc3(h3))
            h3 = self.drop2(h3)
            Experts.append(h3.unsqueeze(0))

            h4 = self.net4(x)
            h4 = h4.view(x.shape[0], -1)
            Experts_feature.append(h4)
            h4 = self.relu(self.fc4(h4))
            h4 = self.drop2(h4)
            Experts.append(h4.unsqueeze(0))

            h5 = self.net5(x)
            h5 = h5.view(x.shape[0], -1)
            Experts_feature.append(h5)
            h5 = self.relu(self.fc5(h5))
            h5 = self.drop2(h5)
            Experts.append(h5.unsqueeze(0))

            h = torch.cat([h_result for h_result in Experts], 0)
            h = torch.sum(h, dim=0).squeeze(0)

        # Get task index for output layer
        if t is None:
            t = 0  # Default to first task
        if isinstance(t, torch.Tensor):
            t = t.item() if t.numel() == 1 else t[0].item()
        
        y = self.last[t](h)

        if return_expert:
            Experts_y = []
            for i in range(self.nExpert):
                h_exp = Experts[i].squeeze(0)
                y_exp = self.last[t](h_exp)
                Experts_y.append(y_exp)
            return y, Experts_y, Experts

        return y

    def mask(self, t, s=1):
        """
        Generate task-adaptive masks for experts.
        
        Args:
            t: Task index (tensor or int)
            s: Scaling factor for gate
        """
        if isinstance(t, torch.Tensor):
            if t.numel() == 1:
                t = t.item()
            else:
                t = t[0].item()
        
        t_tensor = torch.LongTensor([t]).to(next(self.efc1.parameters()).device)
        gfc1 = self.sig_gate(s * self.efc1(t_tensor))
        gfc2 = self.sig_gate(s * self.efc2(t_tensor))
        gfc3 = self.sig_gate(s * self.efc3(t_tensor))
        gfc4 = self.sig_gate(s * self.efc4(t_tensor))
        gfc5 = self.sig_gate(s * self.efc5(t_tensor))
        return [gfc1, gfc2, gfc3, gfc4, gfc5]

    def extract_vector(self, x):
        """Extract feature vector for NME evaluation."""
        if self.use_TG:
            # Use first task for feature extraction
            masks = self.mask(0, s=self.s_gate)
            gfc1, gfc2, gfc3, gfc4, gfc5 = masks
            
            h1 = self.net1(x)
            h1 = h1.view(x.shape[0], -1)
            h1 = self.relu(self.fc1(h1))
            h1 = self.drop2(h1)
            h1 = h1 * gfc1.expand_as(h1)
            
            h2 = self.net2(x)
            h2 = h2.view(x.shape[0], -1)
            h2 = self.relu(self.fc2(h2))
            h2 = self.drop2(h2)
            h2 = h2 * gfc2.expand_as(h2)
            
            h3 = self.net3(x)
            h3 = h3.view(x.shape[0], -1)
            h3 = self.relu(self.fc3(h3))
            h3 = self.drop2(h3)
            h3 = h3 * gfc3.expand_as(h3)
            
            h4 = self.net4(x)
            h4 = h4.view(x.shape[0], -1)
            h4 = self.relu(self.fc4(h4))
            h4 = self.drop2(h4)
            h4 = h4 * gfc4.expand_as(h4)
            
            h5 = self.net5(x)
            h5 = h5.view(x.shape[0], -1)
            h5 = self.relu(self.fc5(h5))
            h5 = self.drop2(h5)
            h5 = h5 * gfc5.expand_as(h5)
            
            h = (h1 + h2 + h3 + h4 + h5) / self.nExpert
        else:
            h1 = self.net1(x)
            h1 = h1.view(x.shape[0], -1)
            h1 = self.relu(self.fc1(h1))
            h1 = self.drop2(h1)
            
            h2 = self.net2(x)
            h2 = h2.view(x.shape[0], -1)
            h2 = self.relu(self.fc2(h2))
            h2 = self.drop2(h2)
            
            h3 = self.net3(x)
            h3 = h3.view(x.shape[0], -1)
            h3 = self.relu(self.fc3(h3))
            h3 = self.drop2(h3)
            
            h4 = self.net4(x)
            h4 = h4.view(x.shape[0], -1)
            h4 = self.relu(self.fc4(h4))
            h4 = self.drop2(h4)
            
            h5 = self.net5(x)
            h5 = h5.view(x.shape[0], -1)
            h5 = self.relu(self.fc5(h5))
            h5 = self.drop2(h5)
            
            h = (h1 + h2 + h3 + h4 + h5) / self.nExpert
        
        return h

