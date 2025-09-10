# RAG 
# pip install markdown PyPDF2 Beautifulsoup4 google-generativeai tiktoken pydantic tqdm 
import tiktoken #OpenAI 分詞器 
import markdown 
import PyPDF2 
import json
import os 
import re

import google.generativeai as genai
import numpy as np

from pydantic import BaseModel #初始化管理套件
from bs4 import BeautifulSoup 
from tqdm import tqdm 

enc = tiktoken.get_encoding("cl100k_base") #100k版 / o200k_base 最新版
'''test str -> token
print(enc.encode(#str))
'''
#tiktoken.encoding_for_model("gpt-4o")

class ReadFiles:   
    """
        將文本切分為帶有重疊的區塊 (chunks)。

        Args:
            text (str): 待切分的原始文本。
            max_token_len (int): 每個區塊的最大 Token 數。
            cover_content (int): 區塊之間重疊的 Token 數。
        
        Returns:
            list[str]: 包含文本區塊的列表。
        """
    def __init__(self,path:str):
        self.path = path
        self.allow_type = ['.md', '.txt' ,'.pdf',]
        self.file_list = self.get_files()
        
    def get_files(self):
        # dir_path夾路徑
        file_list = []
        
        for filepath, dirnames , filenames in os.walk(self.path):
            for i in filenames:
                if any(j in i for j in self.allow_type):
                    file_list.append(os.path.join(filepath, i))
        return file_list
    
    # 讀取 -> 分塊(固定長度) -> 回傳
    def get_content(self,max_token_len:int=600,cover_content:int=150):
        docs = []
        for i in self.file_list:
            content = self.read_file_content(i)
            chunk_content = self.get_chunk(content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs

    @classmethod
    def read_file_content(cls, file_path:str):
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")
    
    @classmethod
    def read_pdf(cls, file_path: str):
        with open(file_path,'rb')as f:
            reader = PyPDF2.RdfReader(f)
            text = ''
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        return text

    @classmethod          
    def read_markdown(cls,file_path:str):
        with open(file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
            html_text = markdown.markdown(md_text)
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            # 使用正則表達式移除網址連接
            text = re.sub(r'http\S+', '', plain_text) 
            return text

    @classmethod
    def read_text(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
        
    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        """
        將文本切分為帶有重疊的區塊 (chunks)。

        Args:
            text (str): 待切分的原始文本。
            max_token_len (int): 每個區塊的最大 Token 數。
            cover_content (int): 區塊之間重疊的 Token 數。
        
        Returns:
            list[str]: 包含文本區塊的列表。
        """
        if max_token_len <= cover_content:
            print("錯誤: max_token_len 必須大於 cover_content")
            return []

        chunk_text = []
        # 將文本按行分割，並移除每行頭尾的空白
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # curr_tokens 用於存放當前區塊的 Token 列表
        curr_tokens = []
        
        # 覆蓋部分的 Token 數量，用於計算新區塊的起點
        # 注意: 這裡為 Token 數量
        overlap_tokens = cover_content 

        for line in lines:
            line_tokens = enc.encode(line)
            
            # 如果當前行加上當前區塊的 Token 數超過了最大長度，則儲存當前區塊
            # +1 是為了考量換行符號
            if len(curr_tokens) + len(line_tokens) + 1 > max_token_len:
                # 將當前區塊的 Token 解碼為字串並儲存
                if curr_tokens:  #  避免加入空 chunk
                  chunk_text.append(enc.decode(curr_tokens))
                
                # 從上一個區塊中截取重疊部分作為新區塊的開頭
                # 這裡的邏輯已經修正為從已儲存的區塊中取 Token 重疊
                overlap_start = max(0, len(curr_tokens) - overlap_tokens)
                curr_tokens = curr_tokens[overlap_start:]
            
            # 將當前行的 Token 添加到當前區塊中
            curr_tokens.extend(line_tokens)
            
            # 確保不會因為換行符號而超過最大長度
            if len(curr_tokens) + 1 <= max_token_len:
                curr_tokens.append(enc.encode('\n')[0])
            
            # 如果某一行本身就比最大長度還長，則直接將該行切分
            # 這裡的邏輯是將長行直接切分成多個帶有重疊的區塊
            if len(curr_tokens) > max_token_len:
                while len(curr_tokens) > max_token_len:
                    # 截取一個完整區塊並加入列表
                    chunk_text.append(enc.decode(curr_tokens[:max_token_len]))
                    
                    # 從剩餘的 Token 中截取重疊部分作為新區塊的開頭
                    curr_tokens = curr_tokens[max_token_len - overlap_tokens:]
        
        # 處理迴圈結束後剩餘的 Token
        if curr_tokens:
            chunk_text.append(enc.decode(curr_tokens))
            
        return chunk_text
'''ReadFiles test bolck
read_from_folder = ReadFiles('#path')
txt_from_files = ReadFiles.get_content(read_from_folder)
print(txt_from_files)
'''

class BaseEmbeddings:
    """
        初始化嵌入
        Args:
            path (str): 模型或數據的路徑
            is_api (bool): True(線上API) / False(本地模型)
    """
    def __init__(self, path: str, is_api: bool):
        self.path = path
        self.is_api = is_api
    def get_embedding(self, text: str, model: str):
        """
        文本->向量
        Args:
            text (str): 輸入文本
            model (str): 使用的模型名稱
        Returns:
            List[float]: 文本的嵌入向量
        Raises:
            NotImplementedError: 需要在子類中實現
        """
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1, vector2) :
        """
        向量間的余弦相似度
        Args:
            vector1 (List[float]): 第一向量
            vector2 (List[float]): 第二向量
        Returns:
            float: 余弦相似度(-1~1)
        """
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)

        # np.isfinite(v1) -> bool 列表
        # np.all 列表全True -> True 
        # NaN, +inf, -inf 會出現False
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            return 0.0
        # 內積
        dot_product = np.dot(v1, v2)
        # 歐氏幾何長度(根號平方和)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        magnitude = norm_v1 * norm_v2
        if magnitude == 0: #不得為0
            return 0.0
            
        # 返回余弦相似度(角度)
        return dot_product / magnitude

# fill def get_embedding
class GoogleEmbedding(BaseEmbeddings):
    def __init__(self, path: str = '', is_api: bool = True, api_key:str=''):
        super().__init__(path, is_api)
        if self.is_api:   
            genai.configure(api_key=api_key)                                
           
    def get_embedding(self, text: str, model_name: str = 'models/gemini-embedding-001'):
        """
        免費嵌入模型 models/gemini-embedding-001 
        """
        if self.is_api:
            text = text.replace("\n", " ")
            result = genai.embed_content(model=model_name, content=text, task_type="RETRIEVAL_DOCUMENT")        
            # 只回傳向量本身，而不是整個回傳物件
            return result['embedding'] 
        else:
            raise NotImplementedError
'''GoogleEmbedding test block   
a = GoogleEmbedding(api_key="your API key",is_api=True)
b = a.get_embedding('這是一個測試用文本。向量化實驗')
print(b)'''

# input: document
class VectorStore:
    def __init__(self, document):
        self.document = document

    #doc轉換成vectors(list)
    def get_vector(self, EmbeddingModel:BaseEmbeddings):
        self.vectors = []
        for doc in tqdm(self.document, desc='Embedding process'):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors
    
    #vectors儲存至document.json
    def persist(self,path:str='storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/document.json','w',encoding='utf-8') as f:
            json.dump(self.document,f,ensure_ascii=False)
        if self.vectors:
            with open(f'{path}/vectors.json','w', encoding='utf-8') as f:
                json.dump(self.vectors,f)

    def load_vector(self,path:str='storage'):
        with open(f'{path}/vectors.json','r',encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f'{path}/document.json','r',encoding='utf-8') as f:
            self.document = json.load(f)
    
    def get_sim(self, vector_1, vector_2):
        return BaseEmbeddings.cosine_similarity(vector_1,vector_2)
    
    #輸入q(文字) => 查找最相似的doc片段 => list
    #vectors : doc的embeding型
    def query (self, q:str, EmbeddingModel:BaseEmbeddings, k:int = 1):
        q_vector = EmbeddingModel.get_embedding(q)
        result = np.array([self.get_sim(q_vector, i) for i in self.vectors])

        return np.array(self.document)[result.argsort()[-k:][::-1] ].tolist()




RAG_PROMPT_TEMPLATE="""
使用以上下文來回答使用者的問題。如果無法回答問題，回答'無法回答問題'。總是使用中文回答。
問題: {question}
可參考的上下文：
···
{context}
···
如果給定的上下文無法讓你做出回答，回答'資料庫中查無資料，故無法回答'。
有用的回答:
"""

'''       
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) #設定key返回bool表示是否成功
'''
api_key="your API key" # <= *這裡因展示功能方便，key不要寫在檔案中*
class GeminiChat(BaseModel):
    def __init__(self, model_name: str = 'gemini-1.5-flash') -> None:  
        super().__init__()
        genai.configure(api_key=api_key)            
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
                

    def chat(self, prompt: str, history, content: str) -> str:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        # Gemini 的 chat 格式
        gemini_history = []
        for msg in history:
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_history.append({'role': role, 'parts': [msg['content']]})
        
        # 準備 RAG 查詢的提示
        rag_prompt = RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)
        gemini_history.append({'role': 'user', 'parts': [rag_prompt]})
        try:
            response = model.generate_content(
                gemini_history,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.1
                )
            )
            return response.text
        except Exception as e:
            # 處理 Gemini 呼叫時可能發生的錯誤
            return f"Error: {e}"
        
if __name__ == "__main__":
    print('-'*10, 'gemini-1.5-flash','-'*10)
    genai.configure(api_key='your API key')
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    question = '2025年8月16日發生什麼大事？' #查詢用文本'8月16日發生甚麼大事' /'誰參加了會議 '
    response = model.generate_content(question)
    print(response.text)
    input('')

    print('-'*10, 'RAG','-'*10)
    doc_route = '文本資料夾的路徑' # <= *指定到資料夾即可，可讀取多個文本檔案*
    docs = ReadFiles(doc_route).get_content(max_token_len=600, cover_content=150)
    print('Chunk Number:',len(docs))
    vector = VectorStore(document=docs)
    embedding = GoogleEmbedding(is_api=True,api_key=api_key)
    #vector.get_vector(EmbeddingModel=embedding)
    #vector.persist(path='storage') #save route

    vector.load_vector('./storage') 
    content = vector.query(q=question, EmbeddingModel=embedding, k=1) #最相關之上下文
    print('Question:',question)
    print('Reference:',content)
  
    chat = GeminiChat()
    print('Response:','\n',chat.chat(prompt=question, history=[], content=content))

        


            
                
