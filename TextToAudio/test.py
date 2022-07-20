import subprocess

subprocess.call('ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi')

print(subprocess.getoutput('nvidia-smi'))