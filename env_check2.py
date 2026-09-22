import sys
print("exe:", sys.executable)
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("mem_gb:", round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))

