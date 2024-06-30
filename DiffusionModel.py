import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, beta_start=0.1, beta_end=0.2, num_diffusion_steps=1000):
        super(DiffusionModel, self).__init__()
        self.num_diffusion_steps = num_diffusion_steps
        
        # Create a schedule for the betas (linearly increasing noise)
        self.beta_schedule = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        self.alpha = 1.0 - self.beta_schedule
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def noise_molecule(self, molecule_repr, time_step):
        # Ensure time_step is properly shaped for batch processing
        if time_step.dim() == 0:
            time_step = time_step.unsqueeze(0)
        
        # Get the noise level for the given time step
        beta_t = self.beta_schedule[time_step]
        alpha_bar_t = self.alpha_bar[time_step]
        
        # Sample Gaussian noise
        noise = torch.randn_like(molecule_repr)
        
        # Calculate the noised molecule representation
        noised_molecule = torch.sqrt(alpha_bar_t).unsqueeze(1) * molecule_repr + torch.sqrt(1 - alpha_bar_t).unsqueeze(1) * noise
        
        return noised_molecule