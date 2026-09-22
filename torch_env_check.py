import torch
with open("D:/unet3/torch_test_result.txt", "w") as f:
    f.write("version=" + torch.__version__ + "\n")
    f.write("cuda=" + str(torch.version.cuda) + "\n")
    f.write("available=" + str(torch.cuda.is_available()) + "\n")
    if torch.cuda.is_available():
        f.write("gpu=" + torch.cuda.get_device_name(0) + "\n")
        f.write("mem_gb=" + str(round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2)) + "\n")
