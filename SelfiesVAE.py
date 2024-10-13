import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# Encoder
# =============================
class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()

        # Convolution layers
        for j in range(params['conv_depth']):
            in_channels = params['NCHARS'] if j == 0 else int(params['conv_dim_depth'] * params['conv_d_growth_factor'] ** (j - 1))
            out_channels = int(params['conv_dim_depth'] * params['conv_d_growth_factor'] ** j)
            kernel_size = int(params['conv_dim_width'] * params['conv_w_growth_factor'] ** j)
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=1))
            if params['batchnorm_conv']:
                self.batchnorm_layers.append(nn.BatchNorm1d(out_channels))

        # Dense (fully connected) layers
        self.fc_layers = nn.ModuleList()
        for i in range(params['middle_layer']):
            input_dim = int(params['hidden_dim'] * params['hg_growth_factor'] ** (params['middle_layer'] - i - 1)) if i == 0 else int(params['hidden_dim'] * params['hg_growth_factor'] ** (params['middle_layer'] - i))
            output_dim = int(params['hidden_dim'] * params['hg_growth_factor'] ** (params['middle_layer'] - i - 1))
            self.fc_layers.append(nn.Linear(input_dim, output_dim))

        self.z_mean = nn.Linear(output_dim, params['hidden_dim'])

    def forward(self, x):
        # Apply convolution layers
        for i, conv in enumerate(self.conv_layers):
            x = F.tanh(conv(x))
            if len(self.batchnorm_layers) > i:
                x = self.batchnorm_layers[i](x)
        x = x.view(x.size(0), -1)  # Flatten

        # Apply fully connected layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        z_mean = self.z_mean(x)

        return z_mean

# =============================
# Decoder
# =============================
class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.fc_layers = nn.ModuleList()
        input_dim = params['hidden_dim']
        for i in range(params['middle_layer']):
            output_dim = int(params['hidden_dim'] * params['hg_growth_factor'] ** i)
            self.fc_layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim
        self.params = params
        self.gru = nn.GRU(input_dim, params['recurrent_dim'], num_layers=params['gru_depth'], batch_first=True)
        self.output_layer = nn.Linear(params['recurrent_dim'], params['NCHARS'])

    def forward(self, z, true_seq=None):
        # Apply fully connected layers
        for fc in self.fc_layers:
            z = F.relu(fc(z))

        z_reps = z.unsqueeze(1).repeat(1, self.params['MAX_LEN'], 1)  # Repeat for GRU

        # Apply GRU
        output, _ = self.gru(z_reps)

        # Final output layer
        output = F.softmax(self.output_layer(output), dim=-1)

        return output

# Encoder Model
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Hard-coded hyperparameters
        self.MAX_LEN = 120   # Adjust based on your dataset
        self.NCHARS = 79     # Number of unique tokens, including special tokens
        self.conv_depth = 4
        self.conv_dim_depth = 8
        self.conv_d_growth_factor = 1.15875438383
        self.conv_dim_width = 8
        self.conv_w_growth_factor = 1.1758149644
        self.batchnorm_conv = False
        self.middle_layer = 1
        self.hg_growth_factor = 1.4928245388
        self.hidden_dim = 100
        self.batchnorm_mid = False
        self.dropout_rate_mid = 0.0

        # Hard-code the input dimension to fully connected layers (calculated manually)
        # This will depend on the convolutional architecture.
        # Example value calculated after the convolutions and flattening
        self.dense_input_dim = 1548  # Example value; adjust according to your actual network architecture

        # Convolution layers
        self.conv_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()

        for j in range(self.conv_depth):
            in_channels = self.NCHARS if j == 0 else int(self.conv_dim_depth * self.conv_d_growth_factor ** (j - 1))
            out_channels = int(self.conv_dim_depth * self.conv_d_growth_factor ** j)
            kernel_size = int(self.conv_dim_width * self.conv_w_growth_factor ** j)
            padding = kernel_size // 2  # To maintain the same length
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
            if self.batchnorm_conv:
                self.batchnorm_layers.append(nn.BatchNorm1d(out_channels))

        # Middle dense layers
        self.fc_layers = nn.ModuleList()
        middle_layer_dims = []
        for i in range(self.middle_layer):
            dim = int(self.hidden_dim * self.hg_growth_factor ** (self.middle_layer - i - 1))
            middle_layer_dims.append(dim)

        for i in range(self.middle_layer):
            input_dim = self.dense_input_dim if i == 0 else middle_layer_dims[i - 1]
            output_dim = middle_layer_dims[i]
            self.fc_layers.append(nn.Linear(input_dim, output_dim))
            if self.batchnorm_mid:
                self.fc_layers.append(nn.BatchNorm1d(output_dim))
            if self.dropout_rate_mid > 0:
                self.fc_layers.append(nn.Dropout(self.dropout_rate_mid))

        # Latent space
        self.z_mean = nn.Linear(middle_layer_dims[-1], self.hidden_dim)
        self.z_log_var = nn.Linear(middle_layer_dims[-1], self.hidden_dim)

    def forward(self, x):
        # x shape: [batch_size, MAX_LEN, NCHARS]
        x = x.permute(0, 2, 1)  # Shape: [batch_size, NCHARS, MAX_LEN]
        for i, conv in enumerate(self.conv_layers):
            x = torch.tanh(conv(x))
            if len(self.batchnorm_layers) > i:
                x = self.batchnorm_layers[i](x)
        x = x.view(x.size(0), -1)  # Flatten the output

        for layer in self.fc_layers:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = torch.tanh(x)
        
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


# Decoder Model
class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        
        # Middle dense layers
        self.fc_layers = nn.ModuleList()
        middle_layer_dims = []
        for i in range(params['middle_layer']):
            dim = int(params['hidden_dim'] * params['hg_growth_factor'] ** i)
            middle_layer_dims.append(dim)
        
        for i in range(params['middle_layer']):
            input_dim = params['hidden_dim'] if i == 0 else middle_layer_dims[i - 1]
            output_dim = middle_layer_dims[i]
            self.fc_layers.append(nn.Linear(input_dim, output_dim))
            if params['batchnorm_mid']:
                self.fc_layers.append(nn.BatchNorm1d(output_dim))
            if params['dropout_rate_mid'] > 0:
                self.fc_layers.append(nn.Dropout(params['dropout_rate_mid']))
        
        # GRU layers
        self.gru = nn.GRU(input_size=middle_layer_dims[-1], hidden_size=params['recurrent_dim'],
                          num_layers=params['gru_depth'], batch_first=True)
        
        # Output layer
        self.output_layer = nn.Linear(params['recurrent_dim'], params['NCHARS'])
        
    def forward(self, z, seq_len):
        # z shape: [batch_size, hidden_dim]
        for layer in self.fc_layers:
            z = layer(z)
            if isinstance(layer, nn.Linear):
                z = torch.tanh(z)
        z = z.unsqueeze(1).repeat(1, seq_len, 1)  # Repeat z for each timestep
        
        # GRU decoding
        output, _ = self.gru(z)
        # Apply dropout if specified
        if self.params['tgru_dropout'] > 0:
            output = nn.functional.dropout(output, p=self.params['tgru_dropout'], training=self.training)
        output = self.output_layer(output)
        return output  # Output logits

# Variational Autoencoder combining Encoder and Decoder
class VAE(nn.Module):
    def __init__(self, params):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(params)
        self.params = params
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std * self.params['kl_loss_weight']
        else:
            return mu
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        seq_len = x.size(1)
        recon_x = self.decoder(z, seq_len)
        return recon_x, mu, logvar