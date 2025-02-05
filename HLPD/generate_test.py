from HLPO import ComputeScore
import torch

import argparse
import numpy as np
import random

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
    parser.add_argument('--task_name', type=str, default="ai_detection_500")
    parser.add_argument('--epochs', type=int, default=2, help="finetuning epochs")
    parser.add_argument('--val_freq', type=int, default=1, help="frequency of eval and saving model")
    parser.add_argument('-ebt', action="store_true", help="Evaluate model before tuning")
    parser.add_argument('--datanum', type=int, default=500, help="num of training data")
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--HLPOtrained', type=str, default="True", choices=["True", "False"], help="If false, means finetuned base model (ablation)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--from_pretrained', type=str)
    parser.add_argument('--eval_dataset', type=str, default="data/rewrite/gpt-4o/xsum_rewrite_gpt-4o.raw_data.json")
    parser.add_argument('--output_file', type=str, default="./results")
    parser.add_argument('--base_model', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--cache_dir', type=str, default="./models")
    args = parser.parse_args()
    
    set_seed(args.seed)
    HLPOtrained = True if args.HLPOtrained == "True" else False
    print(f"Running with args: {args}")
    model = ComputeScore(args.base_model, args.base_model, HLPOtrained=HLPOtrained, HLPO_beta=args.beta, cache_dir=args.cache_dir)
    
    if args.from_pretrained:
        print(f"Loading ckpt from {args.from_pretrained}...")
        model.from_pretrained(args.from_pretrained)
        
    tokenizer= model.scoring_tokenizer
    text = "Chief executive Bimlendra Jha praised the \"significant effort\" to turn things around - after Port Talbot was said to be losing Â"
    tokenized = tokenizer([text], return_tensors="pt").to("cuda")
    outputs = model.scoring_model.generate(**tokenized,     
                                          max_length=200,  # Adjust max length as needed
                                          num_return_sequences=3,
                                          num_beams=10,
                                          early_stopping=True,
                                          no_repeat_ngram_size=2,
                                        )

    for output in outputs:
        predicted_text = tokenizer.decode(output, skip_special_tokens=True)
        print(predicted_text)
        print("------")