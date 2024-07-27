import optuna
import torch
from MolTransformerOptunaTrain import train_and_validate  # Import your main training function from train.py
import os

def objective(trial):
    # Suggest categorical hyperparameters
    d_model = trial.suggest_categorical('d_model', [128, 256])
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    d_ff = trial.suggest_int('d_ff', 512, 4096)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    patience = trial.suggest_int('patience', 3, 10)
    warmup_epochs = trial.suggest_int('warmup_epochs', 0, 40)
    total_epochs = trial.suggest_int('total_epochs', 50, 300)
    # Suggest continuous hyperparameters
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    pretrain_epochs = trial.suggest_int("pretrain_epochs", 0, 40)
    pretrain_learning_rate = trial.suggest_loguniform('pretrain_learning_rate', 1e-5, 1e-2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Call the train_and_validate function with the suggested hyperparameters
    best_val_loss = train_and_validate(d_model, num_heads, num_layers, d_ff, dropout, learning_rate, batch_size, device, warmup_epochs, total_epochs, patience, pretrain_epochs, pretrain_learning_rate, trial)
    return best_val_loss

def main():
    # Delete the existing database file if it exists
    db_file = "smiles_optimization.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    # Use SQLite database to store the study results
    study = optuna.create_study(
        study_name="smiles_optimization",
        storage="sqlite:///smiles_optimization.db",
        direction='minimize',
        pruner=optuna.pruners.MedianPruner()
    )
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
