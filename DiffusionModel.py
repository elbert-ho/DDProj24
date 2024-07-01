import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, unet_model, num_diffusion_steps=1000):
        super(DiffusionModel, self).__init__()
        self.unet_model = unet_model
        self.num_diffusion_steps = num_diffusion_steps

        # Use the cosine beta schedule
        self.beta_schedule = self.cosine_beta_schedule(num_diffusion_steps)
        self.alpha = 1.0 - self.beta_schedule
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]])

        # Learnable variance interpolation parameter
        self.log_var_interpolation = nn.Parameter(torch.zeros(num_diffusion_steps))

    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


    def noise_molecule(self, molecule_repr, time_step, noise=None):
        if time_step.dim() == 0:
            time_step = time_step.unsqueeze(0)
        
        beta_t = self.beta_schedule[time_step]
        alpha_bar_t = self.alpha_bar[time_step]
        
        if noise is None:
            noise = torch.randn_like(molecule_repr)
        
        noised_molecule = torch.sqrt(alpha_bar_t).unsqueeze(1) * molecule_repr + torch.sqrt(1 - alpha_bar_t).unsqueeze(1) * noise
        
        return noised_molecule
    
    def forward(self, molecule_repr, time_step, protein_embedding):
        # Predict the noise
        noise_pred = self.unet_model(molecule_repr, time_step, protein_embedding)
        
        beta_t = self.beta_schedule[time_step]
        log_var = self.log_var_interpolation[time_step] * torch.log(beta_t) + (1 - self.log_var_interpolation[time_step]) * torch.log(torch.tensor([1e-20]))

        return log_var, noise_pred
