# dataset download :
# Pretrain >> 阿里巴巴達摩院 ModelScope: mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2  << This 10G takes time
# SFT >> Hugging Face: BelleGroup/train_3.5M_CN
# pip install modelscope huggingface_hub << run in terminal before using DatasetDownloader

# pre-process :
# mobvoi_seq_monkey_general_open_corpus.jsonl >> seq_monkey_datawhale.jsonl
# train_3.5M_CN.json >> BelleGroup_sft.jsonl

import os
import subprocess
import shutil
import json
from tqdm import tqdm

class DatasetDownloader:
    """
    用於下載和解壓縮 LLM 訓練資料集的類別。
    支援從 ModelScope 和 Hugging Face 下載資料。
    """
    def __init__(self, dataset_base_dir="."):
        """dataset_base_dir (str): 資料集將被下載到的本地基礎目錄。預設為當前目錄。"""
        self.dataset_base_dir = os.path.abspath(dataset_base_dir)
        self._setup_environment()

    def _setup_environment(self):
        """設定必要的環境變數和檢查工具安裝。"""
        # 設定 HF_ENDPOINT 環境變數
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"環境變數 HF_ENDPOINT 已設定為: {os.environ['HF_ENDPOINT']}")

        # 檢查 modelscope 和 huggingface-hub 是否已安裝
        # 這裡僅作簡單檢查，更嚴謹的檢查可以捕捉指令不存在的錯誤
        print("檢查 modelscope 和 huggingface-hub 是否已安裝...")
        try:
            subprocess.run(["pip", "show", "modelscope"], check=True, capture_output=True)
            subprocess.run(["pip", "show", "huggingface-hub"], check=True, capture_output=True)
            print("modelscope 和 huggingface-hub 似乎已安裝。")
        except subprocess.CalledProcessError:
            print("警告：modelscope 或 huggingface-hub 可能未安裝。請嘗試運行 'pip install modelscope huggingface-hub'。")

        # 確保資料集基礎目錄存在
        os.makedirs(self.dataset_base_dir, exist_ok=True)
        print(f"資料集將被下載到: {self.dataset_base_dir}")

    def download_and_extract_modelscope_dataset(self, dataset_name: str, file_name: str):
        """從 ModelScope 下載並解壓縮指定的資料集文件。
            dataset_name (str): ModelScope 上資料集的名稱 (例如: "ddzhu123/seq-monkey")。
            file_name (str): 要下載的資料集文件名 (例如: "mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2")。
        """
        local_folder_path = os.path.join(self.dataset_base_dir, 'tokenizer_data')
        local_file_path = os.path.join(local_folder_path, file_name)
        print(f"\n正在下載 ModelScope 資料集: {dataset_name}/{file_name} 到 {local_folder_path}")

        try:
            # ModelScope 下載指令
            command = [
                "modelscope", "download",
                "--dataset", dataset_name,
                file_name,
                "--local_dir", local_folder_path
            ]
            subprocess.run(command, check=True)
            print(f"ModelScope 資料集 '{file_name}' 下載完成。")

            # 解壓縮資料集
            print(f"正在解壓縮 '{local_file_path}' 到 '{local_folder_path}'...")
            if file_name.endswith(".tar.bz2"):
                # 使用 shutil.unpack_archive 更 Pythonic 且安全
                shutil.unpack_archive(local_file_path, local_folder_path)
                print(f"資料集 '{file_name}' 解壓縮完成。")
            else:
                print(f"無法識別的壓縮格式 '{file_name}'。請手動解壓縮。")

        except subprocess.CalledProcessError as e:
            print(f"下載或解壓縮 ModelScope 資料集時發生錯誤: {e}")
            print(f"錯誤輸出:\n{e.stderr.decode()}")
        except FileNotFoundError:
            print(f"錯誤：'modelscope' 指令未找到。請確認您已安裝 modelscope。")
        except Exception as e:
            print(f"發生未知錯誤: {e}")

    def download_huggingface_dataset(self, repo_id: str, sub_dir: str = ""):
        """ 從 Hugging Face 下載指定的資料集。
            repo_id (str): Hugging Face 上資料集的 repo ID (例如: "BelleGroup/train_3.5M_CN")。
            sub_dir (str): 資料集在本地的子目錄名稱。如果為空，則直接下載到 dataset_base_dir。
                           例如，對於 "BelleGroup/train_3.5M_CN"，可以將 sub_dir 設定為 "BelleGroup"。
        """
        local_target_dir = os.path.join(self.dataset_base_dir, sub_dir) if sub_dir else self.dataset_base_dir
        os.makedirs(local_target_dir, exist_ok=True) # 確保目標目錄存在

        print(f"\n正在下載 Hugging Face 資料集: {repo_id} 到 {local_target_dir}")
        try:
            command = [
                "huggingface-cli", "download",
                "--repo-type", "dataset",
                "--resume-download",
                repo_id,
                "--local-dir", local_target_dir
            ]
            subprocess.run(command, check=True)
            print(f"Hugging Face 資料集 '{repo_id}' 下載完成。")
        except subprocess.CalledProcessError as e:
            print(f"下載 Hugging Face 資料集時發生錯誤: {e}")
            print(f"錯誤輸出:\n{e.stderr.decode()}")
        except FileNotFoundError:
            print(f"錯誤：'huggingface-cli' 指令未找到。請確認您已安裝 huggingface-cli。")
        except Exception as e:
            print(f"發生未知錯誤: {e}")

# 1.處理預訓練文本
def split_text(text, chunk_size=512):
    """將文本按指定長度(512)切分成塊"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def chuck_text_pretain(input_file_path):
    with open('./dataset/tokenizer_data/seq_monkey_datawhale.jsonl', 'a', encoding='utf-8') as pretrain:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in tqdm(data, desc=f"Processing lines in {input_file_path}", leave=False):  # 進度條
                line = json.loads(line)
                text = line['text']
                chunks = split_text(text)
                for chunk in chunks:
                    pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')


# 2.處理SFT文本:
def convert_message(data):
    """將原始數據轉換成標準格式"""
    message = [
        {"role": "system", "content": "你是一個AI助手"}, #system prompt
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

def chuck_text_SFT(input_file_path):
    with open('./dataset/SFT/BelleGroup_sft.jsonl', 'a', encoding='utf-8') as sft:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for item in tqdm(data, desc="Processing", unit="lines"):
                item = json.loads(item)
                message = convert_message(item['conversations'])
                sft.write(json.dumps(message, ensure_ascii=False) + '\n')
#{"conversations": [
#  {"from": "human", "value": "你好，能幫我介紹一下台灣的夜市文化嗎？"},
#  {"from": "assistant", "value": "當然！台灣的夜市是當地獨特的飲食和文化體驗，通常在晚上營業。"}]}
# =>
#[{"role": "system", "content": "你是一個AI助手"},
# {"role": "user", "content": "你好，能幫我介紹一下台灣的夜市文化嗎？"},
# {"role": "assistant", "content": "當然！台灣的夜市是當地獨特的飲食和文化體驗，通常在晚上營業。"}]

# --- 使用範例 ---
if __name__ == "__main__":
    
    # 設定你希望資料集下載到的本地目錄
    # 例如，下載到當前腳本文件夾下的 'downloaded_datasets'
    my_dataset_dir = "./dataset"

    downloader = DatasetDownloader(dataset_base_dir=my_dataset_dir)

    # 下載 SFT 資料集 (Hugging Face)
    downloader.download_huggingface_dataset(
        repo_id="BelleGroup/train_3.5M_CN",
        sub_dir='SFT'  # 子夾設定: my_dataset_dir/.../
    )

    # 下載並解壓縮 ModelScope 資料集
    downloader.download_and_extract_modelscope_dataset(
        dataset_name="ddzhu123/seq-monkey",
        file_name="mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2"
    ) 

    print("\n所有資料集下載和處理已完成。")
    
    
    # 切分
    my_dataset_dir = "./dataset/tokenizer_data"
    print("pre-process begin ...")
    input_file = 'mobvoi_seq_monkey_general_open_corpus.jsonl'
    input_file_path = os.path.join(my_dataset_dir,input_file)   
    chuck_text_pretain(input_file_path)
    print('pretrain dataset finish -- create:seq_monkey_datawhale.jsonl')

    my_dataset_dir = "./dataset/SFT"
    input_file = 'train_3.5M_CN.json'
    input_file_path = os.path.join(my_dataset_dir,input_file)
    chuck_text_SFT(input_file_path)
    print('SFT dataset finish -- create: BelleGroup_sft.jsonl')


