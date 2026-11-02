import torch
model_data = torch.load("dqn_gomoku.pth", map_location=torch.device("cpu"))
print(model_data.keys())
