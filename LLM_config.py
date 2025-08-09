from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from torch.utils.data import Dataset
from typing import Optional

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math
import json

class ModelConfig(PretrainedConfig):
    model_type = 'Tiny-k' #K
    def __init__(
            self,
            dim: int = 768, #維度
            n_layers: int = 12,#層數
            n_heads: int = 16, #頭數
            n_kv_heads: int = 8,#kv頭
            voc_size: int = 6144, #詞表

            hidden_dim: int = 2048, #MLP隱藏層維度(可調)
            multiple_of: int = 64, #(不需要)

            norm_eps: float = 1e-5, #歸一eps
            max_seq_len : int = 512, #最大長度
            dropout: float = 0.0 , #drop機率
            flash_attn :bool =True , #使用flash attension
            **kwargs,
            ) :
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.voc_size = voc_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))#可訓練,初始1
    def _norm(self,x):
        # torch.rsqrt: 平方根的倒數
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+ self.eps)
    def forward(self,x):
        # x -> float -> RMS -> x
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
'''test RMSNorm block
norm = RMSNorm(288,1e-5)
x = torch.randn(1,50,288)
output = norm(x)
print(output.shape)'''

#Grouped-Query Attention，GQA     
# x(a,b,c,d) -> x(a,b,c*n_rep,d) 
def repeat_kv(x:torch.Tensor, n_rep:int):
    b,len,n_kv_heads, head_dim = x.shape
    if n_rep == 1: #不重複
        return x 
    return(
        #添加維度(於head_dim前) -> 擴張為n_rep -> 重排
        x[:,:,:,None,:].expand(b,len,n_kv_heads,n_rep,head_dim).reshape(b,len,n_kv_heads*n_rep,head_dim)
    )

# Potisional Encoding,PE
#(dim,end) -> freqs_cos,freqs_sin(end,dim/2)
#雖叫dim,意義是d/n(每個頭分配的dim)
def precompute_freqs_cis(dim:int, end:int, theta:float =10000.0 ):
    #0~dim,步長2 ->切片 -> float -> /dim -> 作為指數
    #位置編碼通常是將 dim 維度分成兩半，一半用於正弦 (sin) 函數，一半用於餘弦 (cos) 函數
    freqs = 1.0 / (theta **(torch.arange(0,dim,2)[:(dim//2)].float()/dim))#遞減頻率:(dim/2,)
    t = torch.arange(end,device= freqs.device) #位置索引0~end-1, (end,)
    freqs = torch.outer(t,freqs).float() #(end,dim/2)

    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos , freqs_sin

# freqs_cis / x(b,l,n,d/2n) -> freqs_cis(1,l,1,d/2n) 
def reshape_for_broadcast(freqs_cis:torch.Tensor, x:torch.Tensor):
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1],x.shape[-1])

    shape = [d if i==1 or i==(ndim-1) else 1 for i,d in enumerate(x.shape)]
    return freqs_cis.view(shape)

#(xq,xk,freqs_cos,freqs_sin) -> xq_out,xk_out(batch,len,dim/n_head,n_head)
# 要求xq,xk(b,l,n,d/n)
def apply_rotary_emb(xq:torch.Tensor,xk:torch.Tensor,freqs_cos:torch.Tensor,freqs_sin:torch.Tensor):
    #xq(b,l,n,d/n) -> (b,l,n,d/2n,2) -> 2個(b,l,n,d/2n)元組
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1,2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1,2)).unbind(-1)

    #要求:xq_r的第二與最後維度的數字相乘等於(freqs_全元素數量總和)
    #默契:freqs_全元素數量總和=end*dim/2, end:序列長度,而xq_r(b,l,n,d/2n) 
    #因為end==l => dim應為d/n(每個頭分配的dim)
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin #旋後實部
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos #旋後虛部
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    #.stack: [(b,l,n,d/2n),(b,l,n,d/2n)] -> (b,l,n,d/2n,2) 
    #.flatten: (b,l,n,d/2n,2) -> (b,l,n,d/n) 數據為實部虛部交錯排列 ; .cat會是實部排完才虛部
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
'''test RoPE block
xq = torch.randn(1, 50, 6, 48) #(b,l,n,d/n)
xk = torch.randn(1, 50, 6, 48)
#d = 288
cos,sin = precompute_freqs_cis(288//6,50)#(d/n,l)
print(cos.shape)
xq_out,xk_out = apply_rotary_emb(xq,xk,cos,sin)
print(xq_out.shape)'''

#input(b,len,768) ->output(b,len,768)
class Attention(nn.Module):
    def __init__(self, args:ModelConfig):
        super().__init__()
        #根據是否指定n_kv_heads賦值使用GQA(8)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert   args.n_heads % self.n_kv_heads == 0 #總頭/kv頭

        model_parallel_size = 1 #並行處裡大小
        # 本地數量
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads#>1:多頭
        self.head_dim = args.dim // args.n_heads #dim per head
        # QKV
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # output
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        #dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        #rate(default:0)
        self.dropout = args.dropout
        #Flash Attension(torch version > 2.0)
        self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention')
        if not self.flash:
            #不支援Flash,需手動Mask
            print('Slow Attension Mode.(Flash requires Pytorch >=2.0)')
            mask = torch.full( (1,1,args.max_seq_len,args.max_seq_len), float('-inf') )#全填-inf
            mask = torch.triu(mask, diagonal=1)#右上半-inf,其餘0
            self.register_buffer("mask", mask)
    def forward(self,x:torch.Tensor, freqs_cos:torch.Tensor,freqs_sin:torch.Tensor):
        #x(b,len,dim)
        b , seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 調整
        xq = xq.view(b, seq_len, self.n_local_heads, self.head_dim) #->(b,len,16,48)
        xk = xk.view(b, seq_len, self.n_local_kv_heads, self.head_dim) #->(b,len,8,48)
        xv = xv.view(b, seq_len, self.n_local_kv_heads, self.head_dim) #->(b,len,8,48)
        # RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        # GQA
        xk = repeat_kv(xk, self.n_rep) #->(b,len,16,48)
        xv = repeat_kv(xv, self.n_rep)
        # 轉置 for score
        xq = xq.transpose(1, 2) #->(b,16,len,48)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        #Flash Mode
        if self.flash:
            #可設定mask/dropout自動完成
            output = torch.nn.functional.scaled_dot_product_attention(xq,xk,xv,attn_mask=None,dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            #outout(b,16,len,48)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)
        # -> (b,len,16,48) ->(b,len,768)
        output = output.transpose(1, 2).contiguous().view(b, seq_len, -1)

        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

'''#test attension block
args = ModelConfig()
attention_model = Attention(args)
b = 1
seq_len = 50
dim = args.dim
x = torch.rand(b, seq_len, dim)
freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)
output = attention_model(x, freqs_cos, freqs_sin)
print("Output shape:", output.shape)'''

#(b,len,dim) -> (b,len,dim)
# dim:int, hidden_dim:int ,dropout:float
class MLP(nn.Module):
    def __init__(self, dim:int, hidden_dim:int ,dropout:float):
        super().__init__()
        # dim -> hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # hidden_dim -> dim
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # dim -> hidden_dim
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
'''#test MLP block
args = ModelConfig()  
mlp = MLP(args.dim, args.hidden_dim , args.dropout)
x = torch.randn(1, 50, args.dim)
output = mlp(x)
print(output.shape)'''

class DecoderLayer(nn.Module):
    def __init__(self,layer_id:int, args:ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads #16
        self.dim = args.dim #768
        self.head_dim = args.dim // args.n_heads #48
        self.layer_id = layer_id

        # 指定Attention
        self.attention = Attention(args)
        # 指定MLP
        self.feed_forward = MLP(dim=args.dim,hidden_dim=args.hidden_dim, dropout=args.dropout)
        # 歸一化(pre/post)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim,eps=args.norm_eps)
    def forward(self,x,freqs_cos,freqs_sin):
        h = x +self.attention(self.attention_norm(x),freqs_cos,freqs_sin)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

'''# Decoder test block
args = ModelConfig()
decoderlayer = DecoderLayer(0,args)   
b = 1
seq_len = 50
dim = args.dim
x = torch.randn(b, seq_len, dim)
freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)
out = decoderlayer(x, freqs_cos, freqs_sin)
print(out.shape)'''

class Transformer(PreTrainedModel):
    config_class = ModelConfig #
    last_loss: Optional[torch.Tensor] #最後的損失
    def __init__(self, args: ModelConfig):
        super().__init__(args)
        self.args = args
        self.voc_size = args.voc_size
        self.n_layers = args.n_layers
        self.dropout = nn.Dropout(args.dropout)
        # Embedding 
        self.tok_embeddings = nn.Embedding(args.voc_size, args.dim)
        # Decoder
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.voc_size, bias=False)
        #!!Embedding權重語輸出層共享!!
        self.tok_embeddings.weight = self.output.weight

        # RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights) #self-define fn
        # T5 Initialization
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                # w3(MLP) / wo(Attention)
                # dynamic std
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()
        #封裝因果語言模型:past_key_values cache
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens:torch.Tensor, targets:Optional[torch.Tensor]=None,**keyargs):
        #tokens :輸入 / targets:計算損失的答案(訓練時用) / kv_cache: (bool)是否用快取 

        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']
        
        #前向傳播
        b, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # RoPE 切片
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        #Decoder
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin) #12層
        
        h = self.norm(h)

        if targets is not None:#訓練
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
            #內建softmax
        else: #推理
            logits = self.output(h[:, [-1], :]) 
            self.last_loss = None

        # 打包輸出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    #input:idx(b,seqlen) user prompt   
    #temperature : 控制生成文本的隨機性:1.0(標準抽樣) / > 1.0：使概率分佈更平坦 / < 1.0：使概率分佈更尖銳 / 0.0：選擇概率最高的 token
    #top_k: 概率最高的 top_k 個中進行抽樣
    #以下做法 : 在每個生成步驟都重新計算了整個上下文的注意力，導致效率較低
    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        index = idx.shape[1]
        for _ in range(max_new_tokens):#直到至最大長度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            logits = self(idx_cond).logits #self調用forward(tokens=idx_cond)
            logits = logits[:, -1, :] #(b,seqlen,voc) -> (b,voc)

            if temperature == 0.0:#概率最高的那個 token。這種方法生成速度快，但容易陷入重複
                _, idx_next = torch.topk(logits, k=1, dim=-1)  #得分/索引
            else:
                logits = logits / temperature #縮放
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))#若top_k>1,v降冪排列
                    logits[logits < v[:, [-1]]] = -float('Inf') #其餘設為-inf
                probs = F.softmax(logits, dim=-1) #排除-inf
                idx_next = torch.multinomial(probs, num_samples=1)
                #multinomial 函數根據給定的概率分佈進行抽樣

            if idx_next == stop_id:
                break

            # 前文合併
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] # 只返回生成的token
'''Transformer test block
args = ModelConfig()
b = 1
seq_len = 50
x = torch.randint(0, args.voc_size, (b, seq_len)) # [bs, seq_len]
model = Transformer(args=args)
# 計算參數量
num_params = sum(p.numel() for p in model.parameters())
print('Number of parameters:', num_params)
out = model(x)
print(out.logits.shape)'''

#需要指定已配置的tokenizer
#output : X,Y,mask (511,)
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path,'r',encoding = 'utf-8') as f:
            self.data = f.readlines()
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index:int):
        sample = json.loads(self.data[index])
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        #{"id": "doc_1", "text": "這是一個測試。"} >> '<|im_start|>這是一個測試。'
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        #編碼後返回物件 >> 取出數字列表 >> 截斷至最大長度

        #計算填補
        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        #損失遮罩(擋padding)
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64) #截斷最後一字
        Y = np.array(input_id[1:]).astype(np.int64) #不計入第一字
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)#不計入第一字

        return torch.from_numpy(X) , torch.from_numpy(Y) , torch.from_numpy(loss_mask)

#因為使用多輪對話,所以需要區分需要計算損失的位置:從<im_start|>assistant\n後 ~ 第一個|<im_end|>
#output: X,Y,SFT_mask (511,)
class SFTDataset(Dataset):
    def __init__(self,data_path,tokenizer, max_length = 512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0

        with open(data_path,'r',encoding='utf-8') as f :
            self.data = f.readlines()
    def __len__(self):
        return len(self.data)
    
    #用於排除指定序列
    def generate_loss_mask(self,input_ids):
        mask = [0] * len(input_ids)
        a_sequence = self.tokenizer("<|im_start|>assistant\n")['input_ids']  # <|im_start|>assistant\n 查找目標
        a_length = len(a_sequence) #5
        n = len(input_ids)
        i = 0
        while i <= n - a_length:
            match = True
            for k in range(a_length):
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break
            if match:
                # 查找第一個<|im_end|> EOS id
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == self.tokenizer.eos_token_id:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j  # 結束位置j（包含4）
                    # 標記為1（包括start到end）
                    if start <= end:#必成立
                        for pos in range(start, end + 1):
                            if pos < len(mask):#必成立
                                mask[pos] = 1
                # 跳過序列長度，避免重叠匹配
                i += a_length
            else:
                i += 1
        return mask      

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        #計算填補
        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len #[文本數字,0...]長度512
        # mask
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
        
