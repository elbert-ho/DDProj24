import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, unet_model, num_diffusion_steps, device='cuda'):
        super(DiffusionModel, self).__init__()
        self.device = device
        if unet_model is not None:
            self.unet_model = unet_model.to(self.device)
        self.num_diffusion_steps = num_diffusion_steps

        # Use the cosine beta schedule
        self.beta_schedule = self.cosine_beta_schedule(timesteps=num_diffusion_steps, s=.008).to(self.device)
        self.alpha = (1.0 - self.beta_schedule).to(self.device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(self.device)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_bar[:-1]])

        # Learnable variance interpolation parameter
        self.log_var_interpolation = nn.Parameter(torch.zeros(num_diffusion_steps, device=self.device))

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = torch.linspace(1, timesteps + 1, timesteps + 1, device=self.device)
        alphas_cumprod = (torch.cos((steps / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def noise_molecule(self, molecule_repr, time_step, noise=None):
        molecule_repr = molecule_repr.to(self.device)
        time_step = time_step.to(self.device)
        
        if time_step.dim() == 0:
            time_step = time_step.unsqueeze(0)

        beta_t = self.beta_schedule[time_step].to(self.device)
        alpha_bar_t = self.alpha_bar[time_step]
        
        if noise is None:
            noise = torch.randn_like(molecule_repr).to(self.device)
        
        noised_molecule = torch.sqrt(alpha_bar_t).unsqueeze(1) * molecule_repr + torch.sqrt(1 - alpha_bar_t).unsqueeze(1) * noise
        
        return noised_molecule
    
    def forward(self, molecule_repr, time_step, protein_embedding):
        molecule_repr = molecule_repr.to(self.device)
        time_step = time_step.to(self.device)
        protein_embedding = protein_embedding.to(self.device)
        
        # Predict the noise
        noise_pred = self.unet_model(molecule_repr, time_step, protein_embedding)
        beta_t = self.beta_schedule[time_step]
        log_var = None
        # log_var = self.log_var_interpolation[time_step] * torch.log(beta_t) + (1 - self.log_var_interpolation[time_step]) * torch.log(torch.tensor([1e-20], device=self.device))

        return log_var, noise_pred
