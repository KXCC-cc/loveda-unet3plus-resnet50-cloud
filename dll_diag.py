import ctypes, os, sys

torch_lib = r"D:\Anaconda\envs\Buildingfire_Project\Lib\site-packages\torch\lib"
kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
kernel32.LoadLibraryExW.restype = ctypes.c_void_p

# Add DLL directory first
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(torch_lib)

# Try with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
shm = os.path.join(torch_lib, "shm.dll")
c10 = os.path.join(torch_lib, "c10.dll")
torch_cpu = os.path.join(torch_lib, "torch_cpu.dll")
torch_cuda = os.path.join(torch_lib, "torch_cuda.dll")

for dll_path, name in [(c10, "c10.dll"), (torch_cpu, "torch_cpu.dll"), (torch_cuda, "torch_cuda.dll"), (shm, "shm.dll")]:
    res = kernel32.LoadLibraryExW(dll_path, None, 0x00001100)
    err = ctypes.get_last_error()
    sys.stdout.write(f"{name}: result={res}, err={err}\n")
    sys.stdout.flush()

sys.stdout.write("Done\n")
sys.stdout.flush()
