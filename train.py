import torch
import numpy as np
import yaml
from torch import nn, optim
from sklearn.utils.class_weight import compute_class_weight
from galaxy_classification.model.galaxy_cnn import GalaxyClassificationCNNConfig
from galaxy_classification.model.build import build_network
from galaxy_classification.model.config import load_config_from_yaml
from galaxy_classification.galaxy_dataloader import GalaxyDataset, SplitGalaxyClassificationDataSet, SplitGalaxyRegressionDataset
from galaxy_classification.training_utils import fit, compute_epoch_accuracy
from galaxy_classification.model.loss import regression_loss
import shutil
import argparse
import os




parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="no_name_run", help="Directory with resut (in outputs/)")
parser.add_argument("--config", type=str, help="Configuration file for the model and training parameters")
args = parser.parse_args()

# Set device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"TRAINING GALAXY CNN")
print(f"Using device: {device}")
print("---------------------")

#create output directory
run_dir = os.path.join("outputs", args.run_name)
os.makedirs(run_dir, exist_ok=True)



# Load full configuration (CNN + data + training)
with open(args.config, "r") as f:
    full_config = yaml.safe_load(f)

model_conf = full_config["model"]
if model_conf["network_id"] == "regression":
    from galaxy_classification.model.galaxy_cnn import GalaxyRegressionCNNConfig
    cnn_config = GalaxyRegressionCNNConfig(**model_conf)
    mode = "regression"
elif model_conf["network_id"] == "classification":
    cnn_config = GalaxyClassificationCNNConfig(**model_conf)
    mode = "classification"
else:
    raise ValueError("Invalid network_id in config.")
print(f"Network ID: {model_conf['network_id']}")



# Extract data configuration
data_conf = full_config["data"]
data_path = data_conf["image_path"]
label_path = data_conf["label_path"]
input_image_shape = tuple(data_conf["input_image_shape"])
batch_size = data_conf["batch_size"]
val_fraction = data_conf["validation_fraction"]
test_fraction = data_conf["test_fraction"]

# Extract training configuration
train_conf = full_config["training"]
num_epochs = train_conf["num_epochs"]
lr = train_conf["learning_rate"]
weight_decay = train_conf["weight_decay"]
lr = float(train_conf["learning_rate"])
weight_decay = float(train_conf["weight_decay"])

print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print(f"Weight decay: {weight_decay}")
print(f"Number of epochs: {num_epochs}")
print()
print()


# Load dataset
dataset = GalaxyDataset.load(data_path, label_path)
print(f"Number of images: {len(dataset.images)}")

if mode == "classification":
    dataloaders = SplitGalaxyClassificationDataSet(
        dataset,
        batch_size=batch_size,
        validation_fraction=val_fraction,
        test_fraction=test_fraction,
    )
elif mode == "regression":

    dataloaders = SplitGalaxyRegressionDataset(
        dataset,
        batch_size=batch_size,
        validation_fraction=val_fraction,
        test_fraction=test_fraction,
    )

# model
model = build_network(
    input_image_shape=input_image_shape,
    config=cnn_config,
).to(device)

#Loss function
if mode == "classification":
    labels = dataset.labels["label"].values
    present_classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=labels)
    class_weights_full = np.zeros(3)
    for cls, w in zip(present_classes, weights):
        class_weights_full[int(cls)] = w
    class_weights_tensor = torch.tensor(class_weights_full, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
else:
    weight=dataloaders.loss_weights
    def loss_fn(outputs, labels):
        return regression_loss(outputs, labels, weights=weight)
    



# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Train
summary = fit(
    model=model,
    optimizer=optimizer,
    loss_fun=loss_fn,
    train_dataloader=dataloaders.training_dataloader,
    val_dataloader=dataloaders.validation_dataloader,
    num_epochs=num_epochs,
    run_dir=run_dir,
    mode=mode,
    device=device,
)

# ---- Evaluation
if mode == "classification":
    test_accuracy = compute_epoch_accuracy(model, dataloaders.test_dataloader, mode, device)
    print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
    summary.plot_roc_auc(model, dataloaders.test_dataloader, path=run_dir)
else:
    summary.evaluate_regression_metrics(model, dataloaders.test_dataloader, path=run_dir)
    summary.save_predictions(model, dataloaders.test_dataloader)
    summary.plot_true_pred_distributions(path=run_dir)
    
    
    

# ---- Save results
summary.save_plots(run_dir)
summary.save_summary(run_dir)
shutil.copy(args.config, os.path.join(run_dir, "utilized_config.yaml"))
