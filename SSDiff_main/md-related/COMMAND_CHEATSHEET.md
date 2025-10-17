# SSDiff蒸馏命令速查表

快速查找常用命令。

## 🚀 训练命令

### 基础训练

```bash
# 使用脚本（推荐）
bash scripts/train_ssdiff_distill.sh

# 直接命令
CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train_ssdiff_distill.py \
    --pretrained_ssdiff_path /path/to/ssdiff.pt \
    --data_dir /path/to/dataset \
    --output_dir experiments/ssdiff_distill \
    --train_batch_size 4 \
    --learning_rate 5e-5 \
    --max_train_steps 50000 \
    --lora_rank 4
```

### 单GPU训练

```bash
CUDA_VISIBLE_DEVICES="0" python train_ssdiff_distill.py \
    --pretrained_ssdiff_path /path/to/ssdiff.pt \
    --data_dir /path/to/dataset \
    --output_dir experiments/ssdiff_distill \
    --train_batch_size 2 \
    --gradient_accumulation_steps 2
```

### 恢复训练

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train_ssdiff_distill.py \
    --resume_from_checkpoint experiments/ssdiff_distill/checkpoints/model_5000.pkl \
    --resume_step 5000 \
    [其他参数...]
```

## 🧪 测试命令

### 测试原始SSDiff

```bash
# 使用脚本
bash scripts/test_ssdiff_original.sh

# 直接命令 - 10步
python test_ssdiff_unified.py \
    --model_path /path/to/original/ssdiff.pt \
    --use_distillation False \
    --timestep_respacing ddim10 \
    --device cuda:0

# 50步（高质量）
python test_ssdiff_unified.py \
    --model_path /path/to/original/ssdiff.pt \
    --use_distillation False \
    --timestep_respacing ddim50 \
    --device cuda:0
```

### 测试蒸馏SSDiff

```bash
# 使用脚本
bash scripts/test_ssdiff_distilled.sh

# 直接命令
python test_ssdiff_unified.py \
    --model_path experiments/ssdiff_distill/checkpoints/model_10000.pkl \
    --pretrained_ssdiff_path /path/to/original/ssdiff.pt \
    --use_distillation True \
    --timestep_respacing ddim1 \
    --device cuda:0
```

### 速度对比

```bash
# 使用脚本
bash scripts/compare_speed.sh

# 或分别测试
python test_ssdiff_unified.py --use_distillation False --timestep_respacing ddim10 [...]
python test_ssdiff_unified.py --use_distillation True --timestep_respacing ddim1 [...]
```

## 📊 监控命令

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir experiments/ssdiff_distill/logs

# 指定端口
tensorboard --logdir experiments/ssdiff_distill/logs --port 6007

# 后台运行
nohup tensorboard --logdir experiments/ssdiff_distill/logs > tensorboard.log 2>&1 &
```

### 查看训练日志

```bash
# 实时查看
tail -f experiments/ssdiff_distill/logs/train.log

# 查看最后100行
tail -n 100 experiments/ssdiff_distill/logs/train.log

# 搜索错误
grep -i error experiments/ssdiff_distill/logs/train.log
```

## 🔧 调试命令

### 检查环境

```bash
# Python版本
python --version

# PyTorch版本和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# GPU信息
nvidia-smi

# 检查依赖
pip list | grep -E "torch|accelerate|diffusers|peft"
```

### 语法检查

```bash
# 检查Python文件
python -m py_compile ssdiff_distill.py
python -m py_compile train_ssdiff_distill.py
python -m py_compile test_ssdiff_unified.py

# 批量检查
for f in *.py; do python -m py_compile "$f" && echo "✓ $f"; done
```

### 测试导入

```bash
# 测试能否导入模块
python -c "from ssdiff_distill import SSDiff_gen, SSDiff_reg, SSDiff_test"
python -c "import sys; sys.path.append('/data2/user/zelilin/ARConv_SSDiff/SSDiff_main'); from improved_diffusion import logger"
```

## 📦 数据处理

### 检查数据集

```bash
# 查看h5文件信息
python -c "import h5py; f = h5py.File('dataset/test_wv3_multiExm1.h5', 'r'); print(list(f.keys()))"

# 检查数据形状
python dataset-test.py
```

### 结果处理

```bash
# 查看.mat文件
python -c "from scipy.io import loadmat; d = loadmat('test_results/output.mat'); print(d.keys())"

# 统计结果数量
ls test_results/*.mat | wc -l
```

## 🎯 常用参数组合

### 快速实验（少量步数）

```bash
python train_ssdiff_distill.py \
    --max_train_steps 5000 \
    --train_batch_size 2 \
    --lora_rank 2 \
    --checkpointing_steps 100
```

### 高质量训练（推荐）

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train_ssdiff_distill.py \
    --max_train_steps 50000 \
    --train_batch_size 4 \
    --lora_rank 4 \
    --lambda_l2 1.0 \
    --lambda_vsd 1.0 \
    --lambda_vsd_lora 1.0 \
    --mixed_precision fp16
```

### 极致质量（耗时较长）

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch train_ssdiff_distill.py \
    --max_train_steps 100000 \
    --train_batch_size 8 \
    --lora_rank 8 \
    --lambda_l2 2.0 \
    --lambda_vsd 1.5 \
    --learning_rate 3e-5
```

## 💾 文件管理

### 清理临时文件

```bash
# 清理Python缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# 清理日志（谨慎）
rm -rf experiments/*/logs/*

# 清理旧checkpoint（保留最新5个）
cd experiments/ssdiff_distill/checkpoints
ls -t model_*.pkl | tail -n +6 | xargs rm
```

### 备份重要文件

```bash
# 备份checkpoint
cp experiments/ssdiff_distill/checkpoints/model_10000.pkl backup/

# 打包结果
tar -czf results_$(date +%Y%m%d).tar.gz test_results/

# 备份整个实验目录
rsync -av experiments/ssdiff_distill/ backup/ssdiff_distill_$(date +%Y%m%d)/
```

## 🚨 应急命令

### 停止训练

```bash
# 查找进程
ps aux | grep train_ssdiff_distill

# 优雅停止（Ctrl+C或发送SIGINT）
kill -2 <PID>

# 强制停止（不推荐）
kill -9 <PID>
```

### 释放GPU

```bash
# 查看GPU占用
nvidia-smi

# 杀死所有Python进程（危险！）
pkill -9 python

# 杀死特定用户的Python进程
pkill -9 -u $(whoami) python
```

### 快速测试（1张图）

```bash
python test_ssdiff_unified.py \
    --num_test_images 1 \
    --use_distillation True \
    [其他参数...]
```

## 📝 一键命令

### 完整流程（从训练到测试）

```bash
# 1. 训练
bash scripts/train_ssdiff_distill.sh

# 2. 等待训练完成...

# 3. 测试蒸馏模型
bash scripts/test_ssdiff_distilled.sh

# 4. 对比原始模型
bash scripts/test_ssdiff_original.sh

# 5. 速度对比
bash scripts/compare_speed.sh
```

### 批量测试不同checkpoint

```bash
for ckpt in experiments/ssdiff_distill/checkpoints/model_*.pkl; do
    echo "Testing $ckpt"
    python test_ssdiff_unified.py \
        --model_path "$ckpt" \
        --use_distillation True \
        [其他参数...]
done
```

## 🔗 相关命令

### Git操作

```bash
# 查看改动
git status
git diff

# 提交代码
git add .
git commit -m "Add SSDiff distillation implementation"
git push
```

### 环境管理

```bash
# 导出环境
conda env export > environment.yml
pip freeze > requirements.txt

# 重建环境
conda env create -f environment.yml
pip install -r requirements.txt
```

## 📖 帮助命令

```bash
# 查看脚本帮助
python train_ssdiff_distill.py --help
python test_ssdiff_unified.py --help

# 查看所有参数
python train_ssdiff_distill.py --help | grep -E "^\s+--"
```

## 💡 快捷别名（可选）

添加到 `~/.bashrc` 或 `~/.zshrc`:

```bash
# SSDiff蒸馏快捷命令
alias ssdiff-train='bash scripts/train_ssdiff_distill.sh'
alias ssdiff-test-orig='bash scripts/test_ssdiff_original.sh'
alias ssdiff-test-dist='bash scripts/test_ssdiff_distilled.sh'
alias ssdiff-compare='bash scripts/compare_speed.sh'
alias ssdiff-tb='tensorboard --logdir experiments/ssdiff_distill/logs'

# 重载配置
source ~/.bashrc  # 或 source ~/.zshrc
```

使用：

```bash
ssdiff-train      # 开始训练
ssdiff-test-dist  # 测试蒸馏模型
ssdiff-compare    # 速度对比
ssdiff-tb         # 启动TensorBoard
```

---

**提示**: 将常用命令保存到此文件，方便随时查阅！

