@echo off
set PYTHON=D:\Anaconda\envs\Buildingfire_Project\python.exe
echo === 1. Python ===
"%PYTHON%" -c "import sys; print(sys.executable, sys.version)"
echo.
echo === 2. torch ===
"%PYTHON%" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
echo.
echo === 3. deps ===
"%PYTHON%" -c "import numpy, PIL, tqdm; print(numpy.__version__, PIL.__version__, tqdm.__version__)"
pause
