#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import re
import threading
import requests
from huggingface_hub import snapshot_download
from zhipuai import ZhipuAI
import os
from abc import ABC
from ollama import Client
import dashscope
from openai import OpenAI
import numpy as np
import asyncio

from api import settings
from api.utils.file_utils import get_home_cache_dir
from rag.utils import num_tokens_from_string, truncate
import google.generativeai as genai
import json


# 文本嵌入模型模块
# 提供多种文本嵌入服务的实现，用于将文本转换为向量表示

# 基础抽象类，定义了文本嵌入模型的基本接口
class Base(ABC):
    """文本嵌入模型的基础抽象类
    定义了文本嵌入模型的基本接口
    """
    def __init__(self, key, model_name):
        """初始化文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
        """
        pass

    # 将文本列表转换为嵌入向量的方法
    def encode(self, texts: list):
        """将文本列表转换为嵌入向量的方法
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError("Please implement encode method!")

    # 将单个查询文本转换为嵌入向量的方法
    def encode_queries(self, text: str):
        """将单个查询文本转换为嵌入向量的方法
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError("Please implement encode method!")

    # 计算响应中的总token数的方法
    def total_token_count(self, resp):
        """计算响应中的总token数的方法
        Args:
            resp: API响应对象
        Returns:
            总token数
        """
        try:
            return resp.usage.total_tokens
        except Exception:
            pass
        try:
            return resp["usage"]["total_tokens"]
        except Exception:
            pass
        return 0


# 默认的文本嵌入模型实现类，使用FlagEmbedding库
class DefaultEmbedding(Base):
    """默认的文本嵌入模型实现类
    使用FlagEmbedding库进行文本嵌入
    """
    _model = None
    _model_name = ""
    _model_lock = threading.Lock()

    def __init__(self, key, model_name, **kwargs):
        """初始化默认嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        Note:
            如果下载HuggingFace模型遇到问题，可以尝试以下解决方案：
            Linux系统: export HF_ENDPOINT=https://hf-mirror.com
            Windows系统: 祝你好运 ^_-
        """
        if not settings.LIGHTEN:
            with DefaultEmbedding._model_lock:
                from FlagEmbedding import FlagModel
                import torch
                if not DefaultEmbedding._model or model_name != DefaultEmbedding._model_name:
                    try:
                        # 尝试从本地缓存加载模型
                        DefaultEmbedding._model = FlagModel(os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z0-9]+/", "", model_name)),
                                                            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                                            use_fp16=torch.cuda.is_available())
                        DefaultEmbedding._model_name = model_name
                    except Exception:
                        # 如果本地加载失败，从HuggingFace下载模型
                        model_dir = snapshot_download(repo_id="BAAI/bge-large-zh-v1.5",
                                                      local_dir=os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z0-9]+/", "", model_name)),
                                                      local_dir_use_symlinks=False)
                        DefaultEmbedding._model = FlagModel(model_dir,
                                                            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                                            use_fp16=torch.cuda.is_available())
        self._model = DefaultEmbedding._model
        self._model_name = DefaultEmbedding._model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 批量处理文本，每批16个
        batch_size = 16
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        ress = []
        # 分批进行编码
        for i in range(0, len(texts), batch_size):
            ress.extend(self._model.encode(texts[i:i + batch_size]).tolist())
        return np.array(ress), token_count

    def encode_queries(self, text: str):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 计算查询文本的token数
        token_count = num_tokens_from_string(text)
        # 对查询文本进行编码
        return self._model.encode_queries([text]).tolist()[0], token_count


# OpenAI文本嵌入模型实现类
class OpenAIEmbed(Base):
    """OpenAI文本嵌入模型实现类
    使用OpenAI API进行文本嵌入
    """
    def __init__(self, key, model_name="text-embedding-ada-002",
                 base_url="https://api.openai.com/v1"):
        """初始化OpenAI文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # OpenAI要求批量大小不超过16
        batch_size = 16
        # 截断过长的文本
        texts = [truncate(t, 8191) for t in texts]
        ress = []
        total_tokens = 0
        # 分批进行编码
        for i in range(0, len(texts), batch_size):
            res = self.client.embeddings.create(input=texts[i:i + batch_size],
                                                model=self.model_name)
            ress.extend([d.embedding for d in res.data])
            total_tokens += self.total_token_count(res)
        return np.array(ress), total_tokens

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对单个查询文本进行编码
        res = self.client.embeddings.create(input=[truncate(text, 8191)],
                                            model=self.model_name)
        return np.array(res.data[0].embedding), self.total_token_count(res)


# LocalAI本地文本嵌入模型实现类
class LocalAIEmbed(Base):
    """LocalAI本地文本嵌入模型实现类
    使用LocalAI API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url):
        """初始化LocalAI文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        Raises:
            ValueError: 当base_url为空时
        """
        if not base_url:
            raise ValueError("Local embedding model url cannot be None")
        if base_url.split("/")[-1] != "v1":
            base_url = os.path.join(base_url, "v1")
        self.client = OpenAI(api_key="empty", base_url=base_url)
        self.model_name = model_name.split("___")[0]

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        batch_size = 16
        ress = []
        for i in range(0, len(texts), batch_size):
            res = self.client.embeddings.create(input=texts[i:i + batch_size], model=self.model_name)
            ress.extend([d.embedding for d in res.data])
        # 本地嵌入模型不计算token数
        return np.array(ress), 1024

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        embds, cnt = self.encode([text])
        return np.array(embds[0]), cnt


# Azure OpenAI文本嵌入模型实现类
class AzureEmbed(OpenAIEmbed):
    """Azure OpenAI文本嵌入模型实现类
    继承自OpenAIEmbed，使用Azure OpenAI API进行文本嵌入
    """
    def __init__(self, key, model_name, **kwargs):
        """初始化Azure OpenAI文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        from openai.lib.azure import AzureOpenAI
        api_key = json.loads(key).get('api_key', '')
        api_version = json.loads(key).get('api_version', '2024-02-01')
        self.client = AzureOpenAI(api_key=api_key, azure_endpoint=kwargs["base_url"], api_version=api_version)
        self.model_name = model_name


# 百川文本嵌入模型实现类
class BaiChuanEmbed(OpenAIEmbed):
    """百川文本嵌入模型实现类
    继承自OpenAIEmbed，使用百川API进行文本嵌入
    """
    def __init__(self, key,
                 model_name='Baichuan-Text-Embedding',
                 base_url='https://api.baichuan-ai.com/v1'):
        """初始化百川文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if not base_url:
            base_url = "https://api.baichuan-ai.com/v1"
        super().__init__(key, model_name, base_url)


# 通义千问文本嵌入模型实现类
class QWenEmbed(Base):
    """通义千问文本嵌入模型实现类
    使用通义千问API进行文本嵌入
    """
    def __init__(self, key, model_name="text_embedding_v2", **kwargs):
        """初始化通义千问文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        self.key = key
        self.model_name = model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        Raises:
            Exception: 当API调用失败时
        """
        import dashscope
        batch_size = 4
        try:
            res = []
            token_count = 0
            # 截断过长的文本
            texts = [truncate(t, 2048) for t in texts]
            # 分批进行编码
            for i in range(0, len(texts), batch_size):
                resp = dashscope.TextEmbedding.call(
                    model=self.model_name,
                    input=texts[i:i + batch_size],
                    api_key=self.key,
                    text_type="document"
                )
                # 处理返回的嵌入向量
                embds = [[] for _ in range(len(resp["output"]["embeddings"]))]
                for e in resp["output"]["embeddings"]:
                    embds[e["text_index"]] = e["embedding"]
                res.extend(embds)
                token_count += self.total_token_count(resp)
            return np.array(res), token_count
        except Exception as e:
            raise Exception("Account abnormal. Please ensure it's on good standing to use QWen's "+self.model_name)
        return np.array([]), 0

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        Raises:
            Exception: 当API调用失败时
        """
        try:
            # 对查询文本进行编码
            resp = dashscope.TextEmbedding.call(
                model=self.model_name,
                input=text[:2048],
                api_key=self.key,
                text_type="query"
            )
            return np.array(resp["output"]["embeddings"][0]
                            ["embedding"]), self.total_token_count(resp)
        except Exception:
            raise Exception("Account abnormal. Please ensure it's on good standing to use QWen's "+self.model_name)
        return np.array([]), 0


# 智谱AI文本嵌入模型实现类
class ZhipuEmbed(Base):
    """智谱AI文本嵌入模型实现类
    使用智谱AI API进行文本嵌入
    """
    def __init__(self, key, model_name="embedding-2", **kwargs):
        """初始化智谱AI文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        self.client = ZhipuAI(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        arr = []
        tks_num = 0
        MAX_LEN = -1
        # 根据模型类型设置最大长度
        if self.model_name.lower() == "embedding-2":
            MAX_LEN = 512
        if self.model_name.lower() == "embedding-3":
            MAX_LEN = 3072
        if MAX_LEN > 0:
            texts = [truncate(t, MAX_LEN) for t in texts]

        # 逐个文本进行编码
        for txt in texts:
            res = self.client.embeddings.create(input=txt,
                                                model=self.model_name)
            arr.append(res.data[0].embedding)
            tks_num += self.total_token_count(res)
        return np.array(arr), tks_num

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对查询文本进行编码
        res = self.client.embeddings.create(input=text,
                                            model=self.model_name)
        return np.array(res.data[0].embedding), self.total_token_count(res)


# Ollama本地文本嵌入模型实现类
class OllamaEmbed(Base):
    """Ollama文本嵌入模型实现类
    使用Ollama API进行文本嵌入
    """
    def __init__(self, key, model_name, **kwargs):
        """初始化Ollama文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        self.client = Client(host=kwargs.get("base_url", "http://localhost:11434"))
        self.model_name = model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        arr = []
        tks_num = 0
        # 逐个文本进行编码
        for txt in texts:
            res = self.client.embeddings(prompt=txt,
                                         model=self.model_name)
            arr.append(res["embedding"])
            tks_num += 128
        return np.array(arr), tks_num

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对查询文本进行编码
        res = self.client.embeddings(prompt=text,
                                     model=self.model_name)
        return np.array(res["embedding"]), 128


# FastEmbed快速文本嵌入模型实现类
class FastEmbed(DefaultEmbedding):
    """FastEmbed文本嵌入模型实现类
    继承自DefaultEmbedding，使用FastEmbed库进行文本嵌入
    """
    def __init__(
            self,
            key: str | None = None,
            model_name: str = "BAAI/bge-small-en-v1.5",
            cache_dir: str | None = None,
            threads: int | None = None,
            **kwargs,
    ):
        """初始化FastEmbed文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            cache_dir: 缓存目录
            threads: 线程数
            **kwargs: 其他参数
        """
        super().__init__(key, model_name, **kwargs)
        self.cache_dir = cache_dir
        self.threads = threads

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 使用内部tokenizer编码文本并获取总token数
        encodings = self._model.model.tokenizer.encode_batch(texts)
        total_tokens = sum(len(e) for e in encodings)

        # 生成嵌入向量
        embeddings = [e.tolist() for e in self._model.embed(texts, batch_size=16)]

        return np.array(embeddings), total_tokens

    def encode_queries(self, text: str):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 使用内部tokenizer编码文本并获取token数
        encoding = self._model.model.tokenizer.encode(text)
        embedding = next(self._model.query_embed(text)).tolist()

        return np.array(embedding), len(encoding.ids)


# Xinference文本嵌入模型实现类
class XinferenceEmbed(Base):
    """Xinference文本嵌入模型实现类
    使用Xinference API进行文本嵌入
    """
    def __init__(self, key, model_name="", base_url=""):
        """初始化Xinference文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if base_url.split("/")[-1] != "v1":
            base_url = urljoin(base_url, "/v1/embeddings")
        if base_url.find("/embeddings") == -1:
            base_url = urljoin(base_url, "/v1/embeddings")
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
            "Authorization": f"Bearer {key}"
        }

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        data = {
            "model": self.model_name,
            "input": texts
        }
        response = requests.post(self.base_url, headers=self.headers, json=data).json()
        embeddings = []
        for item in response["data"]:
            embeddings.append(item["embedding"])
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对查询文本进行编码
        data = {
            "model": self.model_name,
            "input": [text]
        }
        response = requests.post(self.base_url, headers=self.headers, json=data).json()
        return np.array(response["data"][0]["embedding"]), num_tokens_from_string(text)


# 有道文本嵌入模型实现类
class YoudaoEmbed(Base):
    """有道文本嵌入模型实现类
    使用有道API进行文本嵌入
    """
    _client = None

    def __init__(self, key=None, model_name="maidalun1020/bce-embedding-base_v1", **kwargs):
        """初始化有道文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        if not settings.LIGHTEN and not YoudaoEmbed._client:
            from BCEmbedding import EmbeddingModel
            with threading.Lock():
                if not YoudaoEmbed._client:
                    try:
                        # 尝试从本地加载模型
                        YoudaoEmbed._client = EmbeddingModel(model_name_or_path=os.path.join(
                            get_home_cache_dir(),
                            re.sub(r"^[a-zA-Z0-9]+/", "", model_name)))
                    except Exception:
                        # 如果本地加载失败，从HuggingFace下载
                        YoudaoEmbed._client = EmbeddingModel(
                            model_name_or_path=model_name.replace(
                                "maidalun1020", "InfiniFlow"))
        self._client = YoudaoEmbed._client

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        embeddings = self._client.encode(texts)
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对查询文本进行编码
        embedding = self._client.encode_queries([text])
        return np.array(embedding[0]), num_tokens_from_string(text)


# Jina文本嵌入模型实现类
class JinaEmbed(Base):
    """Jina文本嵌入模型实现类
    使用Jina API进行文本嵌入
    """
    def __init__(self, key, model_name="jina-embeddings-v3",
                 base_url="https://api.jina.ai/v1/embeddings"):
        """初始化Jina文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        self.base_url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        self.model_name = model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 8196) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        data = {
            "model": self.model_name,
            "input": texts
        }
        response = requests.post(self.base_url, headers=self.headers, json=data).json()
        embeddings = []
        for item in response["data"]:
            embeddings.append(item["embedding"])
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        data = {
            "model": self.model_name,
            "input": [text]
        }
        response = requests.post(self.base_url, headers=self.headers, json=data).json()
        return np.array(response["data"][0]["embedding"]), num_tokens_from_string(text)


# Infinity文本嵌入模型实现类
class InfinityEmbed(Base):
    """Infinity文本嵌入模型实现类
    使用Infinity API进行文本嵌入
    """
    _model = None

    def __init__(
            self,
            model_names: list[str] = ("BAAI/bge-small-en-v1.5",),
            engine_kwargs: dict = {},
            key = None,
    ):
        """初始化Infinity文本嵌入模型
        Args:
            model_names: 模型名称列表
            engine_kwargs: 引擎参数
            key: API密钥
        """
        self.model_names = model_names
        self.engine_kwargs = engine_kwargs
        self.key = key

    async def _embed(self, sentences: list[str], model_name: str = ""):
        """异步编码文本
        Args:
            sentences: 文本列表
            model_name: 模型名称
        Returns:
            嵌入向量列表
        """
        # 实现异步编码逻辑
        pass

    def encode(self, texts: list[str], model_name: str = "") -> tuple[np.ndarray, int]:
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
            model_name: 模型名称
        Returns:
            嵌入向量列表和token数量
        """
        # 使用异步引擎进行编码
        embeddings = asyncio.run(self._embed(texts, model_name))
        token_count = sum(num_tokens_from_string(t) for t in texts)
        return np.array(embeddings), token_count

    def encode_queries(self, text: str) -> tuple[np.ndarray, int]:
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对查询文本进行编码
        return self.encode([text])


# Mistral文本嵌入模型实现类
class MistralEmbed(Base):
    """Mistral文本嵌入模型实现类
    使用Mistral API进行文本嵌入
    """
    def __init__(self, key, model_name="mistral-embed",
                 base_url=None):
        """初始化Mistral文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        embeddings = []
        for item in response.data:
            embeddings.append(item.embedding)
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对查询文本进行编码
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text]
        )
        return np.array(response.data[0].embedding), num_tokens_from_string(text)


# AWS Bedrock文本嵌入模型实现类
class BedrockEmbed(Base):
    """Bedrock文本嵌入模型实现类
    使用Bedrock API进行文本嵌入
    """
    def __init__(self, key, model_name,
                 **kwargs):
        """初始化Bedrock文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        import boto3
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=kwargs.get("region_name", "us-east-1"),
            aws_access_key_id=key.split(":")[0],
            aws_secret_access_key=key.split(":")[1]
        )
        self.model_name = model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        embeddings = []
        for text in texts:
            response = self.client.invoke_model(
                modelId=self.model_name,
                body=json.dumps({
                    "inputText": text
                })
            )
            response_body = json.loads(response.get("body").read())
            embeddings.append(response_body["embedding"])
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 调用API进行编码
        response = self.client.invoke_model(
            modelId=self.model_name,
            body=json.dumps({
                "inputText": text
            })
        )
        response_body = json.loads(response.get("body").read())
        return np.array(response_body["embedding"]), num_tokens_from_string(text)


# Google Gemini文本嵌入模型实现类
class GeminiEmbed(Base):
    """Gemini文本嵌入模型实现类
    使用Gemini API进行文本嵌入
    """
    def __init__(self, key, model_name='models/text-embedding-004',
                 **kwargs):
        """初始化Gemini文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        embeddings = []
        for text in texts:
            response = self.model.embed_content(text)
            embeddings.append(response.embedding)
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 配置API密钥
        response = self.model.embed_content(text)
        return np.array(response.embedding), num_tokens_from_string(text)


# NVIDIA文本嵌入模型实现类
class NvidiaEmbed(Base):
    """NVIDIA文本嵌入模型实现类
    使用NVIDIA API进行文本嵌入
    """
    def __init__(
        self, key, model_name, base_url="https://integrate.api.nvidia.com/v1/embeddings"
    ):
        """初始化NVIDIA文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        }

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        payload = {
            "model": self.model_name,
            "input": texts
        }
        response = requests.post(
            self.base_url, json=payload, headers=self.headers
        ).json()
        embeddings = []
        for item in response["data"]:
            embeddings.append(item["embedding"])
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        payload = {
            "model": self.model_name,
            "input": [text]
        }
        response = requests.post(
            self.base_url, json=payload, headers=self.headers
        ).json()
        return np.array(response["data"][0]["embedding"]), num_tokens_from_string(text)


# LM Studio本地文本嵌入模型实现类
class LmStudioEmbed(LocalAIEmbed):
    """LM Studio文本嵌入模型实现类
    继承自LocalAIEmbed，使用LM Studio API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url):
        """初始化LM Studio文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        super().__init__(key, model_name, base_url)


# OpenAI API文本嵌入模型实现类
class OpenAI_APIEmbed(OpenAIEmbed):
    """OpenAI API文本嵌入模型实现类
    继承自OpenAIEmbed，使用OpenAI API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url):
        """初始化OpenAI API文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        super().__init__(key, model_name, base_url)


# CoHere文本嵌入模型实现类
class CoHereEmbed(Base):
    """CoHere文本嵌入模型实现类
    使用CoHere API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url=None):
        """初始化CoHere文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        from cohere import Client
        self.client = Client(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        response = self.client.embed(
            texts=texts,
            model=self.model_name
        )
        return np.array(response.embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对查询文本进行编码
        response = self.client.embed(
            texts=[text],
            model=self.model_name
        )
        return np.array(response.embeddings[0]), num_tokens_from_string(text)


# TogetherAI文本嵌入模型实现类
class TogetherAIEmbed(OpenAIEmbed):
    """TogetherAI文本嵌入模型实现类
    继承自OpenAIEmbed，使用TogetherAI API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url="https://api.together.xyz/v1"):
        """初始化TogetherAI文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        super().__init__(key, model_name, base_url)


# PerfXCloud文本嵌入模型实现类
class PerfXCloudEmbed(OpenAIEmbed):
    """PerfXCloud文本嵌入模型实现类
    继承自OpenAIEmbed，使用PerfXCloud API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url="https://cloud.perfxlab.cn/v1"):
        """初始化PerfXCloud文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        super().__init__(key, model_name, base_url)


# Upstage文本嵌入模型实现类
class UpstageEmbed(OpenAIEmbed):
    """Upstage文本嵌入模型实现类
    继承自OpenAIEmbed，使用Upstage API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url="https://api.upstage.ai/v1/solar"):
        """初始化Upstage文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        super().__init__(key, model_name, base_url)


# SILICONFLOW文本嵌入模型实现类
class SILICONFLOWEmbed(Base):
    """SILICONFLOW文本嵌入模型实现类
    使用SILICONFLOW API进行文本嵌入
    """
    def __init__(
        self, key, model_name, base_url="https://api.siliconflow.cn/v1/embeddings"
    ):
        """初始化SILICONFLOW文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        }

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        payload = {
            "model": self.model_name,
            "input": texts
        }
        response = requests.post(
            self.base_url, json=payload, headers=self.headers
        ).json()
        embeddings = []
        for item in response["data"]:
            embeddings.append(item["embedding"])
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        payload = {
            "model": self.model_name,
            "input": [text]
        }
        response = requests.post(
            self.base_url, json=payload, headers=self.headers
        ).json()
        return np.array(response["data"][0]["embedding"]), num_tokens_from_string(text)


# Replicate文本嵌入模型实现类
class ReplicateEmbed(Base):
    """Replicate文本嵌入模型实现类
    使用Replicate API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url=None):
        """初始化Replicate文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        import replicate
        self.client = replicate.Client(api_token=key)
        self.model_name = model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        embeddings = []
        for text in texts:
            output = self.client.run(
                self.model_name,
                input={"text": text}
            )
            embeddings.append(output)
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对查询文本进行编码
        output = self.client.run(
            self.model_name,
            input={"text": text}
        )
        return np.array(output), num_tokens_from_string(text)


# 百度文心一言文本嵌入模型实现类
class BaiduYiyanEmbed(Base):
    """百度文心一言文本嵌入模型实现类
    使用百度文心一言API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url=None):
        """初始化百度文心一言文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        from qianfan.resources import Embedding
        key = json.loads(key)
        ak = key.get("yiyan_ak", "")
        sk = key.get("yiyan_sk", "")
        self.client = Embedding(ak=ak, sk=sk)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=16):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
            batch_size: 批处理大小
        Returns:
            嵌入向量列表和token数量
        """
        # 对文本列表进行编码
        response = self.client.do(
            texts=texts,
            model=self.model_name
        ).body
        embeddings = []
        for item in response["data"]:
            embeddings.append(item["embedding"])
        token_count = sum(num_tokens_from_string(t) for t in texts)
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对查询文本进行编码
        response = self.client.do(
            texts=[text],
            model=self.model_name
        ).body
        return np.array(response["data"][0]["embedding"]), num_tokens_from_string(text)


# Voyage文本嵌入模型实现类
class VoyageEmbed(Base):
    """Voyage文本嵌入模型实现类
    使用Voyage API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url=None):
        """初始化Voyage文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        import voyageai
        self.client = voyageai.Client(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        response = self.client.embed(
            texts=texts,
            model=self.model_name
        )
        embeddings = []
        for item in response.embeddings:
            embeddings.append(item.embedding)
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对查询文本进行编码
        response = self.client.embed(
            texts=[text],
            model=self.model_name
        )
        return np.array(response.embeddings[0].embedding), num_tokens_from_string(text)


# HuggingFace文本嵌入模型实现类
class HuggingFaceEmbed(Base):
    """HuggingFace文本嵌入模型实现类
    使用HuggingFace API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url=None):
        """初始化HuggingFace文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        self.client = OpenAI(api_key=key, base_url="https://api-inference.huggingface.co/pipeline/feature-extraction")
        self.model_name = model_name

    def encode(self, texts: list):
        """将文本列表转换为嵌入向量
        Args:
            texts: 文本列表
        Returns:
            嵌入向量列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 2048) for t in texts]
        # 计算总token数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行编码
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        embeddings = []
        for item in response.data:
            embeddings.append(item.embedding)
        return np.array(embeddings), token_count

    def encode_queries(self, text):
        """将单个查询文本转换为嵌入向量
        Args:
            text: 查询文本
        Returns:
            嵌入向量和token数量
        """
        # 对查询文本进行编码
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text]
        )
        return np.array(response.data[0].embedding), num_tokens_from_string(text)


# 火山引擎文本嵌入模型实现类
class VolcEngineEmbed(OpenAIEmbed):
    """火山引擎文本嵌入模型实现类
    继承自OpenAIEmbed，使用火山引擎API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url="https://ark.cn-beijing.volces.com/api/v3"):
        """初始化火山引擎文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        super().__init__(key, model_name, base_url)


# GPUStack文本嵌入模型实现类
class GPUStackEmbed(OpenAIEmbed):
    """GPUStack文本嵌入模型实现类
    继承自OpenAIEmbed，使用GPUStack API进行文本嵌入
    """
    def __init__(self, key, model_name, base_url):
        """初始化GPUStack文本嵌入模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        super().__init__(key, model_name, base_url)