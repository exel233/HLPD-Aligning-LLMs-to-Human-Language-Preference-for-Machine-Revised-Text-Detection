from dataset import CustomDataset_split, CustomDataset_rewrite
from HLPO import ComputeScore
from engine import run
import torch
from torch.utils.data import Subset
import argparse
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('-a', type=int, default=1, help="accumulation steps")
    parser.add_argument('--task_name', type=str, default="ai_detection_3000")
    parser.add_argument('--epochs', type=int, default=2, help="finetuning epochs")
    parser.add_argument('--val_freq', type=int, default=1, help="frequency of eval and saving model")
    parser.add_argument('-ebt', action="store_true", help="Evaluate model before tuning")
    parser.add_argument('--datanum', type=int, default=3000, help="num of training data")
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--HLPOtrained', type=str, default="True", choices=["True", "False"], help="If false, means finetuned base model (ablation)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--from_pretrained', type=str)
    parser.add_argument('--eval_dataset', type=str, default="data/rewrite/gpt-4o/xsum_rewrite_gpt-4o.raw_data.json")
    parser.add_argument('--output_file', type=str, default="./results")
    parser.add_argument('--base_model', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--cache_dir', type=str, default="./models")
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('-- min_delta', type=float, default=0.03)
    args = parser.parse_args()
    
    set_seed(args.seed)
    HLPOtrained = True if args.HLPOtrained == "True" else False
    print(f"Running with args: {args}")
    model = ComputeScore(args.base_model, args.base_model, HLPOtrained=HLPOtrained, HLPO_beta=args.beta, cache_dir=args.cache_dir)
    
    if args.from_pretrained:
        print(f"Loading ckpt from {args.from_pretrained}...")
        model.from_pretrained(args.from_pretrained)
        
    train_data = CustomDataset_split(data_json_dir='data/ai_detection_3000_polish.raw_data.json',val_ratio=0) 
    val_data = CustomDataset_rewrite(data_json_dir=args.eval_dataset)
    
    if not os.path.exists(args.output_file):
        os.makedirs(args.output_file)
    subset_indices = torch.randperm(len(train_data))[:args.datanum]
    
    with open("indice.txt","w+") as f:
        f.write(str(subset_indices.tolist()))
        
    train_subset = Subset(train_data, subset_indices)
    print(len(train_subset))
    print(len(val_data))
    
    run(
        model, 
        [train_subset, val_data], 
        DEVICE='cuda', 
        ckpt_dir=f"./ckpt/{args.task_name}_HLPO_lr_{args.lr}_beta_{args.beta}_a_{args.a}",
        args=args
    )