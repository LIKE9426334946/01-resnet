# 01-resnet

# 使用虚拟环境
E:\project\myooo\.venv\Scripts\Activate.ps1

# ddp执行命令
torchrun --standalone --nproc-per-node=2 testcifar10.py  
或者  
python -m torch.distributed.run --standalone --nproc-per-node=2 testcifar10.py  