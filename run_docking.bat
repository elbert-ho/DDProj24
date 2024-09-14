@echo off

:: Commands to run in the Windows environment
echo Running Windows commands...
:: First run DockingTest
python DockingTest.py
:: Clear out both folders
python ClearFolders.py
python 

echo Done with Windows commands.

:: Switch to WSL, activate Conda environment, and run Linux commands
echo Running WSL commands...

wsl bash -c "source ~/.bashrc && conda activate myenv && echo 'Conda environment activated' && ls -la"

echo Done with WSL commands.

pause
