import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Device name:", torch.cuda.get_device_name(0))
    print("Number of CUDA devices:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
