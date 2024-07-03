import optuna
import torch
from MolTransformerOptunaTrain import train_and_validate  # Import your main training function from train.py

def objective(trial):
    # Suggest hyperparameters
    num_heads = trial.suggest_int('num_heads', 4, 12)
    
    # Ensure d_model is a power of two times num_heads
    valid_d_models = [num_heads * (2 ** i) for i in range(3, 6)] 
    d_model = trial.suggest_categorical('d_model', valid_d_models)
    
    num_layers = trial.suggest_int('num_layers', 2, 8)
    d_ff = trial.suggest_int('d_ff', 512, 4096)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Call the train_and_validate function with the suggested hyperparameters
    best_val_loss = train_and_validate(d_model, num_heads, num_layers, d_ff, dropout, learning_rate, batch_size, device)
    return best_val_loss

def main():
    # Use SQLite database to store the study results
    study = optuna.create_study(study_name="smiles_optimization", storage="sqlite:///smiles_optimization.db", direction='minimize')
    # Run the optimization
    study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed

    # Print the best trial
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()
