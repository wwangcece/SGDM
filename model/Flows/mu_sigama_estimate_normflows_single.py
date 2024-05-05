# Import required packages
import os
import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

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
    mean_npz_file = 'model/Flows/results/style_mean.npy'
    std_npz_file = 'model/Flows/results/style_std.npy'
    model_save_path = "model/Flows/checkpoints"
    training_loss_save_path = 'model/Flows/results/loss.png'
    device = 'cuda:1'
    total_epoch_num = 1000

    # Create a dataset
    dataset = Dataset(mean_npz_file, std_npz_file)

    # Create a data loader
    dataloader = DataLoader(dataset, batch_size=1000, num_workers=16)

    # Set up model
    # Define 2D Gaussian base distribution
    model_mean = CreateFlow(dim=32, num_layers=16, hidden_layers=[16, 64, 64, 32])
    model_std = CreateFlow(dim=32, num_layers=16, hidden_layers=[16, 64, 64, 32])

    # Move model on GPU if available
    device = torch.device(device)
    model_mean = model_mean.to(device)
    model_std = model_std.to(device)

    # Train model
    optimizer_mean = torch.optim.Adam(model_mean.parameters(), lr=4e-4, weight_decay=1e-5)
    optimizer_std = torch.optim.Adam(model_std.parameters(), lr=4e-4, weight_decay=1e-5)
    scheduler_mean = StepLR(optimizer_mean, step_size=20000, gamma=0.5)
    scheduler_std = StepLR(optimizer_std, step_size=20000, gamma=0.5)

    iter = 0
    loss_mean_dist = np.array([])
    loss_std_dist = np.array([])
    for epoch in range(total_epoch_num):
        for batch in dataloader:
            optimizer_mean.zero_grad()
            optimizer_std.zero_grad()
            
            # Get training samples
            target_mean, target_std = batch
            target_mean = target_mean.to(device)
            target_std = target_std.to(device)
            
            # Compute loss
            loss_mean = model_mean.forward_kld(target_mean)
            loss_std = model_std.forward_kld(target_std)

            # Do backprop and optimizer step
            if ~(torch.isnan(loss_mean) | torch.isinf(loss_mean)):
                loss_mean.backward()
                optimizer_mean.step()
                scheduler_mean.step()
                loss_mean_dist = np.append(loss_mean_dist, loss_mean.to('cpu').data.numpy())
            if ~(torch.isnan(loss_std) | torch.isinf(loss_std)):
                loss_std.backward()
                optimizer_std.step()
                scheduler_std.step()
                loss_std_dist = np.append(loss_std_dist, loss_std.to('cpu').data.numpy())
            if (iter) % 10 == 0:
                print("iter{} mean loss:{}  std loss:{} lr:{}".format(iter, loss_mean.to('cpu').data.numpy(), loss_std.to('cpu').data.numpy(), optimizer_mean.param_groups[0]['lr']))

            iter += 1
    
    plt.figure(figsize=(10, 10))
    plt.plot(loss_mean_dist, label='mean_loss')
    plt.plot(loss_std_dist, label='std_loss')
    plt.legend()
    plt.savefig(training_loss_save_path)

    # 保存模型
    torch.save(model_mean.state_dict(), os.path.join(model_save_path, "flow_mean.pt"))
    torch.save(model_std.state_dict(), os.path.join(model_save_path, "flow_std.pt"))
    print("model checkpoints saved")

if __name__ == "__main__":
    main()
