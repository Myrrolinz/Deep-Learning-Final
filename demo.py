import torch

x = torch.tensor([
    [[0.3000, 0.2926],[-0.2705, -0.2632]],
    [[-0.1821, -0.1747],[-0.1526, -0.1453]],
    [[-0.0642, -0.0568],[-0.0347, -0.0274]],
    [[-0.0642, -0.0568],[-0.0347, -0.0274]]
])

print(len(x))
print("1",torch.norm(x, p=2,dim=1))

print("2",torch.max(x, 1)[0].unsqueeze(1))

t=torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )
print("3",t.shape)

pool_type=["1","2"]
print("pool_type size:",len(pool_type))