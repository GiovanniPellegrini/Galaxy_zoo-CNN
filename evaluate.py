import argparse
import os
import torch
from galaxy_classification.galaxy_dataloader import GalaxyDataset, SplitGalaxyDataLoader
from galaxy_classification.model.galaxy_cnn import GalaxyClassificationCNNConfig
import yaml
from galaxy_classification.model.build import build_network
from galaxy_classification.training_utils import compute_epoch_accuracy, tta, TrainingSummary

parser=argparse.ArgumentParser(description="Evaluate a trained model")


parser.add_argument("--model_directory", type=str, help="Directory with model to test (in outputs/)")
parser.add_argument("--run_name", type=str, help="Directory where test data will be stored (in model_name/)")
parser.add_argument("--tta", action="store_true", help="Use test time augmentation")
args=parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"EVA GALAXY CNN")
print(f"Using device: {device}")
print("---------------------")

#create the directory for the test data
directory_path = os.path.join("outputs", args.run_name)
os.makedirs(directory_path, exist_ok=True)

model_directory_path=os.path.join("outputs", args.model_directory)
model_path=os.path.join(model_directory_path, "best_model.pth")
config_path=os.path.join(model_directory_path, "config.yaml")
with open(config_path, "r") as f:
    full_config = yaml.safe_load(f)
model_config = GalaxyClassificationCNNConfig(**full_config["model"])
data_conf = full_config["data"]


dataset=GalaxyDataset.load( image_path=data_conf["image_path"],label_path=data_conf["label_path"])
dataloaders = SplitGalaxyDataLoader(
    dataset,
    batch_size=data_conf["batch_size"],
    validation_fraction=data_conf["validation_fraction"],
    test_fraction=data_conf["test_fraction"]
)

#model
model=build_network(
    input_image_shape=data_conf["input_image_shape"],
    config=model_config
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

#test
test_dataloader=dataloaders.test_dataloader

if args.tta:
    print("Using test time augmentation")
    correct=0
    total=0
    for batch in test_dataloader:
        inputs = batch["images"]
        labels = batch["labels"].to(device)
        outputs = tta(model, inputs, device, n=5)
        predicted_classes = outputs.argmax(dim=1)
        correct += (predicted_classes == labels).sum().item()
        total += len(labels)
    test_accuracy = correct / total
else:
    test_accuracy = compute_epoch_accuracy(model, test_dataloader, device)
print(f"Test accuracy: {test_accuracy:.4f}")       

#summary = TrainingSummary(interval=1)
#summary.save_roc_auc(model, dataloaders.test_dataloader, device, path=directory_path)
#summary.save_plots(directory_path)