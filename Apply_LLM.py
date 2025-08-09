# Generate text:
from transformers import AutoTokenizer, AutoModelForCausalLM
from LLM_config import ModelConfig, Transformer
from contextlib import nullcontext

import argparse
import pickle
import torch
import os 

class TextGenerator:
    def __init__(self,
                 model_path = './models/SFT/SFT_1024_18_6144_final.pth',
                 tokenizer_path='./tokenizer',
                 seed = 42,
                 device = None,
                 dtype = 'bfloat16'): #'float32'/'float16'
        '''初始化'''
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.seed = seed
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.model = Transformer(ModelConfig(dim=1024,n_layer=18))
        
        # Setting
        torch.manual_seed(seed) #CPU
        torch.cuda.manual_seed(seed) #GPU
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        device_type = "cuda" if "cuda" in self.device else "cpu"
        the_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device == "cpu" else torch.amp.autocast(device_type=device_type, dtype=the_dtype)

        model_dict = torch.load(self.model_path, map_location=self.device)
        sunwanted_prefix = '_orig_mod.'
        for k, v in list(model_dict.items()):
            if k.startswith(sunwanted_prefix):
                model_dict[k[len(sunwanted_prefix):]] = model_dict.pop(k)
        self.model.load_state_dict(model_dict, strict=False)

        #參數量計算 sum(p.numel() for p in model.parameters())/ sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"LLM總參數量: {num_params / 1e6:.3f} 百萬")

        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def chat_template(self, prompt):
        message = [
            {"role": "system", "content": "你是一個AI助手"},#system prompt
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    #回傳generated_texts
    def sft_sample(self, 
               start="Hello!",  # 生成文本的起始提示詞(任意字符串)
               num_samples=3,  # 生成樣本的數量(default = 3)
               max_new_tokens=256,  # 最大token數(default = 256)
               temperature=1.0,  # 隨機性，1.0標準，值越大越随機
               top_k=300): #機率最高的幾個
        
        start = self.chat_template(start)
        start_ids = self.tokenizer(start).data['input_ids']
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]) #轉乘tensor,再增加batch的維度->(b,len)

        generated_texts = [] #生成的樣本
        with torch.inference_mode():
            with self.ctx:
                for k in range(num_samples):
                    y = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
                    generated_texts.append(self.tokenizer.decode(y[0].tolist()) ) # (b,len) -> (len,) -> 轉成list

        return generated_texts
    
    #回傳generated_texts
    def pretrain_sample(self, 
               start="Hello!",  # 生成文本的起始提示詞(任意字符串)
               num_samples=3,  # 生成樣本的數量(default = 3)
               max_new_tokens=256,  # 最大token數(default = 256)
               temperature=0.7,  # 隨機性，1.0標準，值越大越随機
               top_k=300): #機率最高的幾個
        
        if start.startswith('FILE:'): #'FILE:(route)'
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()
        start_ids = self.tokenizer(start).data['input_ids']
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])
        generated_texts = []
        with torch.inference_mode():
            with self.ctx:
                for k in range(num_samples):
                    y = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
                    generated_texts.append(self.tokenizer.decode(y[0].tolist()) ) # (b,len) -> (len,) -> 轉成list

        return generated_texts

if __name__ == '__main__':
    print('='*10,'Inference Mode : Pre_train model','='*10)

    pretrain_prompt_datas = [
        '<|im_start|>北京大学是什么?',
        '<|im_start|>台灣位於?',
    ]
 #./models/SFT/SFT_1024_18_6144_final.pth / ./models/pretrain/pretrain_1024_18_6144_final.pth
    generator = TextGenerator(model_path='./models/pretrain/pretrain_1024_18_6144_final.pth')
    for i in range(len(pretrain_prompt_datas)):
        samples = generator.pretrain_sample(start=pretrain_prompt_datas[i], num_samples=1, max_new_tokens=120, temperature=0.75)
        print(f"Question {i+1}:")
        print(f'{pretrain_prompt_datas[i]}')
        print(f'Answer {i+1}:')
        print(f'{samples[0]}')
        print('-'*20)
        
  
    print('='*10,'Inference Mode : SFT model','='*10)

    SFT_prompt_datas = [
        '北京大学是什么?',
        '台灣位於?',
    ]
    generator = TextGenerator(model_path='./models/SFT/SFT_1024_18_6144_final.pth')
    for i in range(len(SFT_prompt_datas)):
        samples = generator.pretrain_sample(start=SFT_prompt_datas[i], num_samples=1, max_new_tokens=120, temperature=0.75)
        print(f"Question {i+1}:")
        print(f'{SFT_prompt_datas[i]}')
        print(f'Answer {i+1}:')
        print(f'{samples[0]}')
        print('-'*20)

    #外部輸入方法、無限對話、上文記憶
    





