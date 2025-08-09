# BPE Tokenizer
import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (decoders,models,pre_tokenizers,trainers,Tokenizer,)
from tokenizers.normalizers import NFKC
from typing import Generator

def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """讀取JSON文本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data['text'] #逐行生成器
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                print(e)
                continue

#建立 : tokenizer_config.json / special_tokens_map.json
def create_tokenizer_config(save_dir: str) -> None:
    """設定"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 建立special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)   

#建立 : tokenizer.json / tokenizer_config.json / special_tokens_map.json
#上列檔案為.from_pretrained約定黨名: 分詞規則與詞表 / 分詞器類別與設定 / 特殊詞元
def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """訓練並保存tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()  # 添加文本規範化:不同形式但語義相同的字元轉換為統一的形式，如全形半形轉換
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)#預分詞器，在BPE之前分割成更小的單元
    tokenizer.decoder = decoders.ByteLevel()#與預分詞器對應

    # 配置特殊token
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|im_start|>", 
        "<|im_end|>"
    ]

    # 訓練器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,  # 提高低頻過濾:濾掉訓練數據中的罕見錯誤或噪音
        show_progress=True, #顯示設定
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()#初始詞彙表中的字符集
    )

    # 訓練tokenizer
    print(f"Training tokenizer with data from {data_path}")
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))

    # 驗證特殊token映射
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error:", e)
        raise

    # 保存tokenizer文件:tokenizer.json
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    # 建立配置文件
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")        

#測試功能
def eval_tokenizer(tokenizer_path: str) -> None:
    """tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 基本屬性
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # 聊天模板
    messages = [
        {"role": "system", "content": "你是一位AI助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    
    print("\n=== 聊天模板測試 ===")
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        # add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    # 解碼測試
    print("\n=== 解碼測試 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # 特殊token處理
    print("\n=== 特殊token處理 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)

if __name__ == '__main__':
    # 配置路径
    data_path = "./dataset/tokenizer_data/mobvoi_seq_monkey_general_open_corpus.jsonl"
    save_dir = "./tokenizer"
    # 訓練tokenizer
    
    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=6144)

    # 評估tokenizer
    eval_tokenizer(save_dir)