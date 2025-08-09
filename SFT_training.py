# SFT training: (almost the same with LLM_training.py) I use !! to mark differences
# Dataset -> SFTDataset
# pre-load model

from LLM_config import Transformer,SFTDataset,ModelConfig #!!
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from contextlib import nullcontext
import torch.optim as optim

import argparse
import datetime
import torch
import math
import time
import os

#input:it,all >> output:lr
def get_lr(it, all):
    """
    學習率計算: 余弦退火策略
    1. Warmup: 從0線性增長到目标
    2. 余弦退火: 按余弦函数衰减到最小
    3. 超出訓練步数后: 保持最小    
    Args:
        it (int): 當前迭代步數
        all (int): 總迭代步數       
    Returns:
        float: 當前學習率
    """
    warmup_iters = args.warmup_iters  # 預熱數
    lr_decay_iters = all  
    min_lr = args.learning_rate / 10  # 最小學習率:0.1初始

    # Warmup：
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # 超出訓練步数：保持最小学习率
    if it > lr_decay_iters:
        return min_lr
    
    # 余弦退火:
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 余弦係數
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """
    訓練流程:
        1.數據讀取 / 設備轉移
        2.學習率調整
        3.前向傳播 / 損失計算
        4.梯度累積 / 反向傳播
        5.梯度剪裁 / 優化器更新
        6.輸出紀錄 / 儲存模型
    Args:
        epoch(int) :當前回合數
    """
    
    # 1.
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        if step < args.start_batch:
            continue
        start_time = time.time()
        print('[','*'*int(step*20//iter_per_epoch),'-'*int(20 -(step*20//iter_per_epoch)),']',end='\r')
        X = X.to(args.device)  # 輸入序列
        Y = Y.to(args.device)  # 目標序列
        loss_mask = loss_mask.to(args.device) #訓練版mask:遮蓋padding

        # 2.
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #混合精度訓練:
        with ctx:
            # 3.
            out = model(X,Y)
            loss = out.last_loss / args.accumulation_steps
            #apply mask
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()
        # 4.使用scaler混合精度反向傳播
        scaler.scale(loss).backward()
        # 5.每固定batch後剪裁
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)#取消縮放
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) #剪裁
            scaler.step(optimizer)
            scaler.update() #更新
            optimizer.zero_grad(set_to_none=True) #梯度歸零:set_to_none=True可以節省内存

            #6.  每百batch顯示一次
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            print(f"[Epoch:[{epoch + 1}/{args.epochs}]({step}/{iter_per_epoch})]"
                  f"[Loss:{loss.item() * args.accumulation_steps:.3f}]"
                  f"[LR:{optimizer.param_groups[-1]['lr']:.7f}]"
                  f"[Time left for finish epoch:{ datetime.timedelta(seconds=spend_time  * (iter_per_epoch-(step+1) ) )}];")#預計剩餘時間
            
        if (step + 1) % args.save_interval == 0:
            model.eval()
            # 帶步数的檢查點
            ckp = f'{args.out_dir}/SFT_{lm_config.dim}_{lm_config.n_layers}_{lm_config.voc_size}_step{step+1}.pth'
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()

def init_model():
    """
    初始化模型和分詞器:
        1. 讀取tokenizer
        2. 建立Transformer模型
        3. 設置GPU並行訓練(if)
        4. 指定設備
        5. 統計/列印参數量
    Returns:
        tuple: (model, tokenizer)
    """
    def count_parameters(model):
        """
        統計可訓練參數        
        Args:
            model: PyTorch模型           
        Returns:
            int: 可訓練參數總量
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer/')#讀取3個json
    model = Transformer(lm_config)
    num_gpus = torch.cuda.device_count() #多卡計算
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel!")
        # 使用DataParallel包裝模型以支持多GPU訓練
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)
    print(f'LLM總參數量: {count_parameters(model) / 1e6:.3f} 百萬')
    return model, tokenizer

if __name__ == "__main__":
    # ==================== 參數包 ====================
    parser = argparse.ArgumentParser(description="Tiny-LLM SFT")
    parser.add_argument("--start_batch", type=int, default=180000,help="batch to start training from")
    parser.add_argument("--start_epoch", type=int, default=-1,help="epoch to start training from")


    parser.add_argument("--out_dir", type=str, default="./models/SFT", help="模型輸出目錄") #!!
    parser.add_argument("--data_path", type=str, default="./dataset/SFT/BelleGroup_sft.jsonl", help="訓練數據路徑")#!!
    parser.add_argument("--epochs", type=int, default=2, help="訓練輪數")
    parser.add_argument("--batch_size", type=int, default=16, help="batch大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="學習率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="設備")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="數據類別")
    parser.add_argument("--num_workers", type=int, default=0, help="進程數")
    #優化參數
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累積步數")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度剪裁閾值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="預熱數")
    #保存參數
    parser.add_argument("--log_interval", type=int, default=1000, help="記錄間隔")
    parser.add_argument("--save_interval", type=int, default=10000, help="保存間隔")

    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="使用的GPU ID:用逗號分隔 (例如: '0,1,2')")
    args = parser.parse_args()

    # 設置GPU:
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # 第一個可用GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # 配置類別:
    lm_config = ModelConfig(dim=1024,n_layers=18,)   # 更改:維度dim / 層數

    # 訓練環境:
    max_seq_len = lm_config.max_seq_len #512
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)
    # CPU用nullcontext，GPU用autocast
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type)

    model, tokenizer = init_model()

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len) #!!
    train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,  # 64
            pin_memory=True,             # 讀取到固定内存中，加速GPU傳輸
            drop_last=False,             # 不丢棄最後一個不完整的批次
            shuffle=True,                # 随機打亂數據
            num_workers=args.num_workers # 讀取進程數
        )
    
    # 初始化混合精度訓練的梯度縮放器
    scaler = torch.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 優化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # ==================== 開始訓練 ====================
    iter_per_epoch = len(train_loader) #batch數量, 數量=總量/大小
    print('Settings Verified:',f'{args.device}' , '-> SFT start...')
    print('Batch number per epoch:',f'{iter_per_epoch}')

    # 讀取pre-train model or SFT checkpoint!!
    ckp = f'./models/SFT/SFT_1024_18_6144_step{args.start_batch}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    unwanted_prefix = '_orig_mod.' #去除特定表頭
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    print('Loading Pre-train model finish--- ')

    for epoch in range(args.epochs):
        if epoch < args.start_epoch:
            continue
        train_epoch(epoch)

    model.eval()
    # 帶步数的檢查點
    ckp = f'{args.out_dir}/SFT_{lm_config.dim}_{lm_config.n_layers}_{lm_config.voc_size}_final.pth'
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state_dict, ckp)
    model.train()

    print('SFT Finished---')
