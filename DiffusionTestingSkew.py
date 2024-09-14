from DiffusionModelGLIDE import *
from unet_condition import Text2ImUNet
import yaml
import matplotlib.pyplot as plt

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

n_diff_step = config["diffusion_model"]["num_diffusion_steps"]
batch_size = config["diffusion_model"]["batch_size"]
protein_embedding_dim = config["protein_model"]["protein_embedding_dim"]
lr = config["diffusion_model"]["lr"]
# epochs = config["diffusion_model"]["epochs"]
patience = config["diffusion_model"]["patience"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diffusion_model = GaussianDiffusion(betas=get_named_beta_schedule(n_diff_step))
unet = Text2ImUNet(text_ctx=1, xf_width=protein_embedding_dim, xf_layers=0, xf_heads=0, xf_final_ln=0, tokenizer=None, in_channels=1, model_channels=128, out_channels=2, num_res_blocks=2, attention_resolutions=[], dropout=.1, channel_mult=(1, 2, 4, 8), dims=1)
unet.load_state_dict(torch.load('unet_resized_skew2.pt', map_location=device))
unet.to(device)

prot = torch.zeros([1, 3], device="cuda")

frequency = 10
amplitude = 1
phase = math.pi / 4

prot[0][0] = frequency
prot[0][1] = amplitude
prot[0][2] = phase
sample = diffusion_model.p_sample_loop(unet, (1, 1, 128), prot=prot, w=0).reshape(128).detach().cpu().numpy()
plt.plot(sample)

real_wave = np.linspace(0, 2 * np.pi, 128)
sine_wave = amplitude * np.sin(frequency* real_wave + phase)
plt.plot(sine_wave)

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Y Coordinate')
plt.title('Plot of Y Coordinates')

# Show legend
plt.legend()

# Show the plot
plt.show()