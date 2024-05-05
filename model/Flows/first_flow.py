import torch
import flowtorch.bijectors as bij
import flowtorch.distributions as dist
import matplotlib.pyplot as plt

device = "cuda:2"

# initial distribution
base_dist = torch.distributions.Independent(
    torch.distributions.Normal(
        (torch.zeros(2) + 8).to(device), (torch.ones(2) * 5).to(device)
    ),
    1,
)
# target distribution
target_dist = torch.distributions.Independent(
    torch.distributions.Normal(
        (torch.zeros(2) + 10).to(device), (torch.ones(2) * 5).to(device)
    ),
    1,
)

# Lazily instantiated flow
bijectors = bij.AffineAutoregressive()
# parameterized transfer function/estimated distribution(flow)
# (knowing a reversible transfer function, we could calculate its probability distribution function conditioned on the base distribution)
flow = dist.Flow(base_dist, bijectors).to(device)

# Training loop
opt = torch.optim.Adam(flow.parameters(), lr=5e-3)
for idx in range(10001):
    opt.zero_grad()

    # Minimize KL(p || q)
    y = target_dist.sample((1000,)).to(device)
    # different from other networks, only backpropagation is done during training for flow networks
    # optimize parameters to ensure that samples from target distribution could get as high score as possible in the estimated distribution
    loss = -flow.log_prob(y).mean()

    if idx % 500 == 0:
        print("epoch", idx, "loss", loss.item())

    loss.backward()
    opt.step()

# 保存模型
output_path = "./checkpoints/FirstFlow.pth"
torch.save(flow.state_dict(), output_path)

# 加载模型
loaded_flow = dist.Flow(base_dist, bijectors)
loaded_flow.load_state_dict(torch.load(output_path))
loaded_flow.to(device)
loaded_flow.eval()

# 执行推理
with torch.no_grad():
    samples = loaded_flow.sample((1000,)).cpu().detach().numpy()
    y = target_dist.sample((1000,)).cpu().detach().numpy()

plt.scatter(samples[:, 0], samples[:, 1], label="Inference Result")
plt.scatter(y[:, 0], y[:, 1], label="Target Distribution")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()

outFigPath = "./flowResults.png"
plt.savefig(outFigPath)
plt.show()
