# Import required packages
import os
import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# to train a normalizing flow: prevent the model from being overfitted!!!
# (1) model size ~ training dataset size
# (2) training epoch
# (3) learning rate

# set up dataset
class Dataset(Dataset):
    def __init__(self, mean_npz_file, std_npz_file):
        mean_data = np.load(mean_npz_file)
        std_data = np.load(std_npz_file)
        self.mean_data = torch.from_numpy(mean_data)  # 'data' is the key in the npz file
        self.std_data = torch.from_numpy(std_data)

    def __len__(self):
        return len(self.mean_data)

    def __getitem__(self, idx):
        return self.mean_data[idx], self.std_data[idx]

def CreateFlow(dim, num_layers=32, hidden_layers=[1, 64, 64, 2], init_zeros=True, permute_mode="shuffle"):
    base = nf.distributions.base.DiagGaussian(dim)
    # Define list of flows
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        # AffineCouplingBlock: split-AffineCoupling-merge; so input-dim is half of base-dist-dim; out-dim is just base-dist-dim(scale and shift both involved)
        param_map = nf.nets.MLP(hidden_layers, init_zeros=init_zeros)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map="sigmoid"))
        # Swap dimensions
        flows.append(nf.flows.Permute(dim, mode=permute_mode))
    # Construct flow model
    model = nf.NormalizingFlow(base, flows)

    return model

def main():
    # Replace 'your_npz_file.npz' with the actual file path
    mean_npz_file = 'model/Flows/results/mean.npy'
    std_npz_file = 'model/Flows/results/std.npy'
    val_mean = "model/Flows/results/val_mean.npy"
    val_std = "model/Flows/results/val_std.npy"
    model_save_path = "model/Flows/checkpoints"
    print_iter = 10
    val_iter = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create a dataset
    train_dataset = Dataset(mean_npz_file, std_npz_file)
    val_dataset = Dataset(val_mean, val_std)

    # Create a data loader
    train_dataloader = DataLoader(train_dataset, batch_size=5000, num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=396, num_workers=16)

    # Set up model
    # Define 2D Gaussian base distribution
    model = CreateFlow(dim=512, num_layers=4, hidden_layers=[256, 512])

    # Move model on GPU if available
    model = model.to(device)

    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    iter = 0
    train_loss_dist = np.array([])
    val_loss_dist = np.array([])
    for epoch in range(500):
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Get training samples
            target_mean, target_std = batch
            target = torch.concat((target_mean, target_std), dim=1).to(device)
            
            # Compute loss
            loss = model.forward_kld(target)

            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
                train_loss_dist = np.append(train_loss_dist, loss.to('cpu').data.numpy())

            if (iter) % print_iter == 0:
                print("iter{} training_loss:{}".format(iter, loss.to('cpu').data.numpy()))
            if (iter) % val_iter == 0:
                with torch.no_grad():
                    val_loss = 0
                    for batch in val_dataloader:
                        target_mean, target_std = batch
                        target = torch.concat((target_mean, target_std), dim=1).to(device)
                        val_loss += model.forward_kld(target).to('cpu').data.numpy()
                    print("iter{} validation_loss:{}".format(iter, val_loss))
                    val_loss_dist = np.append(val_loss_dist, val_loss)

            iter += 1
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(model_save_path, "flow_mean_std_mini_1.1"))
    print("model checkpoints saved")

    plt.figure(figsize=(10, 10))
    plt.plot(train_loss_dist, label='train_loss')
    plt.plot(val_loss_dist, label='val_loss')

    plt.legend()
    plt.savefig("model/Flows/loss.png")

if __name__ == "__main__":
    main()
