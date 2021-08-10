import torch
import torch.nn as nn

class TemperatureScalingModel(nn.Module):
    def __init__(self, model, device):
        nn.Module.__init__(self)
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, X):
        logits = self.model(X)
        return self.temperature_scale(logits)

    def temperature_scale(self, all_logits):
        return all_logits.to(self.device) / self.temperature.unsqueeze(1).expand(all_logits.size()).to(self.device)

    def caribrate(self, all_logits, all_labels):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.0001, max_iter=500)
        nll_criterion = nn.CrossEntropyLoss()
        def eval():
            loss = nll_criterion(self.temperature_scale(all_logits), all_labels)
            loss.backward()
            return loss
        optimizer.step(eval)

