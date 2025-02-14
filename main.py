import torch
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer
import argparse
import yaml
from ViT import VisionTransformer
from data_preparation import prepare_data

class ModelConfig:
    def __init__(self, hyperparams):
        self.batch_size = int(hyperparams.get('batch_size', 16))
        self.image_size = int(hyperparams.get('image_size', 32))
        self.patch_size = int(hyperparams.get('patch_size', 8))
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.n_embd = int(hyperparams.get('n_embd', 32))
        self.hidden_size = int(hyperparams.get('hidden_size', 64))
        self.num_classes = int(hyperparams.get('num_classes', 10)) #CIFAR-10
        self.num_channels = int(hyperparams.get('num_channels', 3))
        self.bias = bool(hyperparams.get('bias', True))
        self.epochs = int(hyperparams.get('epochs', 100))
        self.dropout = float(hyperparams.get('dropout', 0)) #Default No droput
        self.learning_rate = float(hyperparams.get('learning_rate', 1e-3))
        self.weight_decay = float(hyperparams.get('weight_decay', 0.01)) #default value of Adam
        self.beta1 = float(hyperparams.get('beta1', 0.9)) #default value of Adam
        self.beta2 = float(hyperparams.get('beta2', 0.99)) #default value of Adam
        self.n_head = int(hyperparams.get('n_head', 4))
        self.n_layer = int(hyperparams.get('n_layer', 4))
        self.mlp_expansion_ratio = int(hyperparams.get('mlp_expansion_ratio', 4))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-model-every", type=int, default=0)

    args = parser.parse_args()
    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {args.device}")
    return args


def main():
    args = parse_args()
    with open("hyperparameters.yaml") as f:
        hyperparams = yaml.load(f, Loader=yaml.FullLoader)
    device = args.device
    save_model_every_n_epochs = args.save_model_every
    config = ModelConfig(hyperparams)
    trainloader, testloader, _ = prepare_data(batch_size=config.batch_size)
    model = VisionTransformer(config)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, device=device, config=config)
    trainer.train(trainloader, testloader, config.epochs, save_model_every_n_epochs=save_model_every_n_epochs)


if __name__ == "__main__":
    main()