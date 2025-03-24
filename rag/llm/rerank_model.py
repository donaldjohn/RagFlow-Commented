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
import re
import threading
from urllib.parse import urljoin

import requests
import httpx
from huggingface_hub import snapshot_download
import os
from abc import ABC
import numpy as np
from yarl import URL

from api import settings
from api.utils.file_utils import get_home_cache_dir
from rag.utils import num_tokens_from_string, truncate
import json


# sigmoid函数，用于将分数归一化到0-1之间
def sigmoid(x):
    """sigmoid函数，用于将分数归一化到0-1之间
    Args:
        x: 输入值
    Returns:
        归一化后的值
    """
    return 1 / (1 + np.exp(-x))


# 重排序模型模块
# 提供多种重排序服务的实现，用于对检索结果进行重新排序，提高检索准确性

# 基础抽象类，定义了重排序模型的基本接口
class Base(ABC):
    """重排序模型的基础抽象类
    定义了重排序模型的基本接口
    """
    def __init__(self, key, model_name):
        """初始化重排序模型
        Args:
            key: API密钥
            model_name: 模型名称
        """
        pass

    # 计算查询文本与文档列表的相似度分数
    def similarity(self, query: str, texts: list):
        """计算查询文本与文档列表的相似度分数
        Args:
            query: 查询文本
            texts: 文档列表
        Returns:
            相似度分数列表和token数量
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError("Please implement encode method!")

    # 计算响应中的总token数
    def total_token_count(self, resp):
        """计算响应中的总token数
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


# 默认的重排序模型实现类，使用FlagEmbedding库
class DefaultRerank(Base):
    """默认的重排序模型实现类
    使用FlagEmbedding库进行重排序
    """
    _model = None
    _model_lock = threading.Lock()

    def __init__(self, key, model_name, **kwargs):
        """
        初始化默认重排序模型
        如果下载HuggingFace模型遇到问题，可以尝试以下解决方案：

        Linux系统:
        export HF_ENDPOINT=https://hf-mirror.com

        Windows系统:
        祝你好运 ^_-
        """
        if not settings.LIGHTEN and not DefaultRerank._model:
            import torch
            from FlagEmbedding import FlagReranker
            with DefaultRerank._model_lock:
                if not DefaultRerank._model:
                    try:
                        # 尝试从本地加载模型
                        DefaultRerank._model = FlagReranker(
                            os.path.join(get_home_cache_dir(), re.sub(r"^[a-zA-Z0-9]+/", "", model_name)),
                            use_fp16=torch.cuda.is_available())
                    except Exception:
                        # 如果本地加载失败，从HuggingFace下载模型
                        model_dir = snapshot_download(repo_id=model_name,
                                                      local_dir=os.path.join(get_home_cache_dir(),
                                                                             re.sub(r"^[a-zA-Z0-9]+/", "", model_name)),
                                                      local_dir_use_symlinks=False)
                        DefaultRerank._model = FlagReranker(model_dir, use_fp16=torch.cuda.is_available())
        self._model = DefaultRerank._model

    def similarity(self, query: str, texts: list):
        """计算查询文本与文档列表的相似度分数
        Args:
            query: 查询文本
            texts: 文档列表
        Returns:
            相似度分数列表和token数量
        """
        # 构建查询-文档对
        pairs = [(query, truncate(t, 2048)) for t in texts]
        token_count = 0
        for _, t in pairs:
            token_count += num_tokens_from_string(t)
        batch_size = 4096
        res = []
        # 分批计算相似度分数
        for i in range(0, len(pairs), batch_size):
            scores = self._model.compute_score(pairs[i:i + batch_size], max_length=2048)
            # 使用sigmoid函数将分数归一化到0-1之间
            scores = sigmoid(np.array(scores)).tolist()
            if isinstance(scores, float):
                res.append(scores)
            else:
                res.extend(scores)
        return np.array(res), token_count


# Jina重排序模型实现类
class JinaRerank(Base):
    """Jina重排序模型实现类
    使用Jina API进行重排序
    """
    def __init__(self, key, model_name="jina-reranker-v2-base-multilingual",
                 base_url="https://api.jina.ai/v1/rerank"):
        """初始化Jina重排序模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        self.base_url = "https://api.jina.ai/v1/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        self.model_name = model_name

    def similarity(self, query: str, texts: list):
        """计算查询文本与文档列表的相似度分数
        Args:
            query: 查询文本
            texts: 文档列表
        Returns:
            相似度分数列表和token数量
        """
        # 截断过长的文本
        texts = [truncate(t, 8196) for t in texts]
        data = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts)
        }
        # 调用API进行重排序
        res = requests.post(self.base_url, headers=self.headers, json=data).json()
        rank = np.zeros(len(texts), dtype=float)
        for d in res["results"]:
            rank[d["index"]] = d["relevance_score"]
        return rank, self.total_token_count(res)


# 有道重排序模型实现类
class YoudaoRerank(DefaultRerank):
    """有道重排序模型实现类
    继承自DefaultRerank，使用BCEmbedding库进行重排序
    """
    _model = None
    _model_lock = threading.Lock()

    def __init__(self, key=None, model_name="maidalun1020/bce-reranker-base_v1", **kwargs):
        """初始化有道重排序模型
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        if not settings.LIGHTEN and not YoudaoRerank._model:
            from BCEmbedding import RerankerModel
            with YoudaoRerank._model_lock:
                if not YoudaoRerank._model:
                    try:
                        # 尝试从本地加载模型
                        YoudaoRerank._model = RerankerModel(model_name_or_path=os.path.join(
                            get_home_cache_dir(),
                            re.sub(r"^[a-zA-Z0-9]+/", "", model_name)))
                    except Exception:
                        # 如果本地加载失败，从HuggingFace下载
                        YoudaoRerank._model = RerankerModel(
                            model_name_or_path=model_name.replace(
                                "maidalun1020", "InfiniFlow"))

        self._model = YoudaoRerank._model

    def similarity(self, query: str, texts: list):
        """计算查询文本与文档列表的相似度分数
        Args:
            query: 查询文本
            texts: 文档列表
        Returns:
            相似度分数列表和token数量
        """
        # 构建查询-文档对
        pairs = [(query, truncate(t, self._model.max_length)) for t in texts]
        token_count = 0
        for _, t in pairs:
            token_count += num_tokens_from_string(t)
        batch_size = 8
        res = []
        # 分批计算相似度分数
        for i in range(0, len(pairs), batch_size):
            scores = self._model.compute_score(pairs[i:i + batch_size], max_length=self._model.max_length)
            # 使用sigmoid函数将分数归一化到0-1之间
            scores = sigmoid(np.array(scores)).tolist()
            if isinstance(scores, float):
                res.append(scores)
            else:
                res.extend(scores)
        return np.array(res), token_count


# Xinference重排序模型实现类
class XInferenceRerank(Base):
    """Xinference重排序模型实现类
    使用Xinference API进行重排序
    """
    def __init__(self, key="xxxxxxx", model_name="", base_url=""):
        """初始化Xinference重排序模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if base_url.find("/v1") == -1:
            base_url = urljoin(base_url, "/v1/rerank")
        if base_url.find("/rerank") == -1:
            base_url = urljoin(base_url, "/v1/rerank")
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
            "Authorization": f"Bearer {key}"
        }

    def similarity(self, query: str, texts: list):
        """计算查询文本与文档列表的相似度分数
        Args:
            query: 查询文本
            texts: 文档列表
        Returns:
            相似度分数列表和token数量
        """
        if len(texts) == 0:
            return np.array([]), 0
        # 构建查询-文档对
        pairs = [(query, truncate(t, 4096)) for t in texts]
        token_count = 0
        for _, t in pairs:
            token_count += num_tokens_from_string(t)
        data = {
            "model": self.model_name,
            "query": query,
            "return_documents": "true",
            "return_len": "true",
            "documents": texts
        }
        # 调用API进行重排序
        res = requests.post(self.base_url, headers=self.headers, json=data).json()
        rank = np.zeros(len(texts), dtype=float)
        for d in res["results"]:
            rank[d["index"]] = d["relevance_score"]
        return rank, token_count


# LocalAI本地重排序模型实现类
class LocalAIRerank(Base):
    """LocalAI本地重排序模型实现类
    使用LocalAI API进行重排序
    """
    def __init__(self, key, model_name, base_url):
        """初始化LocalAI重排序模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if base_url.find("/rerank") == -1:
            self.base_url = urljoin(base_url, "/rerank")
        else:
            self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        self.model_name = model_name.split("___")[0]

    def similarity(self, query: str, texts: list):
        """计算查询文本与文档列表的相似度分数
        Args:
            query: 查询文本
            texts: 文档列表
        Returns:
            相似度分数列表和token数量
        Raises:
            ValueError: 当API响应不包含结果时
        """
        # 截断过长的文本
        texts = [truncate(t, 500) for t in texts]
        data = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行重排序
        res = requests.post(self.base_url, headers=self.headers, json=data).json()
        rank = np.zeros(len(texts), dtype=float)
        if 'results' not in res:
            raise ValueError("response not contains results\n" + str(res))
        for d in res["results"]:
            rank[d["index"]] = d["relevance_score"]

        # 将分数归一化到0-1之间
        min_rank = np.min(rank)
        max_rank = np.max(rank)

        # 避免除零错误
        if max_rank - min_rank != 0:
            rank = (rank - min_rank) / (max_rank - min_rank)
        else:
            rank = np.zeros_like(rank)

        return rank, token_count


# NVIDIA重排序模型实现类
class NvidiaRerank(Base):
    """NVIDIA重排序模型实现类
    使用NVIDIA API进行重排序
    """
    def __init__(
            self, key, model_name, base_url="https://ai.api.nvidia.com/v1/retrieval/nvidia/"
    ):
        """初始化NVIDIA重排序模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if not base_url:
            base_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
        self.model_name = model_name

        # 根据模型名称设置不同的API端点
        if self.model_name == "nvidia/nv-rerankqa-mistral-4b-v3":
            self.base_url = os.path.join(
                base_url, "nv-rerankqa-mistral-4b-v3", "reranking"
            )

        if self.model_name == "nvidia/rerank-qa-mistral-4b":
            self.base_url = os.path.join(base_url, "reranking")
            self.model_name = "nv-rerank-qa-mistral-4b:1"

        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

    def similarity(self, query: str, texts: list):
        """计算查询文本与文档列表的相似度分数
        Args:
            query: 查询文本
            texts: 文档列表
        Returns:
            相似度分数列表和token数量
        """
        token_count = num_tokens_from_string(query) + sum(
            [num_tokens_from_string(t) for t in texts]
        )
        data = {
            "model": self.model_name,
            "query": {"text": query},
            "passages": [{"text": text} for text in texts],
            "truncate": "END",
            "top_n": len(texts),
        }
        # 调用API进行重排序
        res = requests.post(self.base_url, headers=self.headers, json=data).json()
        rank = np.zeros(len(texts), dtype=float)
        for d in res["rankings"]:
            rank[d["index"]] = d["logit"]
        return rank, token_count


# LM Studio本地重排序模型实现类
class LmStudioRerank(Base):
    """LM Studio本地重排序模型实现类
    目前未实现
    """
    def __init__(self, key, model_name, base_url):
        """初始化LM Studio重排序模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        pass

    def similarity(self, query: str, texts: list):
        """计算查询文本与文档列表的相似度分数
        Args:
            query: 查询文本
            texts: 文档列表
        Returns:
            相似度分数列表和token数量
        Raises:
            NotImplementedError: 因为此功能尚未实现
        """
        raise NotImplementedError("The LmStudioRerank has not been implement")


# OpenAI API重排序模型实现类
class OpenAI_APIRerank(Base):
    """OpenAI API重排序模型实现类
    使用OpenAI API进行重排序
    """
    def __init__(self, key, model_name, base_url):
        """初始化OpenAI重排序模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if base_url.find("/rerank") == -1:
            self.base_url = urljoin(base_url, "/rerank")
        else:
            self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        self.model_name = model_name.split("___")[0]

    def similarity(self, query: str, texts: list):
        """计算查询文本与文档列表的相似度分数
        Args:
            query: 查询文本
            texts: 文档列表
        Returns:
            相似度分数列表和token数量
        Raises:
            ValueError: 当API响应不包含结果时
        """
        # 截断过长的文本
        texts = [truncate(t, 500) for t in texts]
        data = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        # 调用API进行重排序
        res = requests.post(self.base_url, headers=self.headers, json=data).json()
        rank = np.zeros(len(texts), dtype=float)
        if 'results' not in res:
            raise ValueError("response not contains results\n" + str(res))
        for d in res["results"]:
            rank[d["index"]] = d["relevance_score"]

        # 将分数归一化到0-1之间
        min_rank = np.min(rank)
        max_rank = np.max(rank)

        # 避免除零错误
        if max_rank - min_rank != 0:
            rank = (rank - min_rank) / (max_rank - min_rank)
        else:
            rank = np.zeros_like(rank)

        return rank, token_count


# CoHere重排序模型实现类
class CoHereRerank(Base):
    def __init__(self, key, model_name, base_url=None):
        from cohere import Client

        self.client = Client(api_key=key)
        self.model_name = model_name

    def similarity(self, query: str, texts: list):
        token_count = num_tokens_from_string(query) + sum(
            [num_tokens_from_string(t) for t in texts]
        )
        # 调用API进行重排序
        res = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=texts,
            top_n=len(texts),
            return_documents=False,
        )
        rank = np.zeros(len(texts), dtype=float)
        for d in res.results:
            rank[d.index] = d.relevance_score
        return rank, token_count


# TogetherAI重排序模型实现类
class TogetherAIRerank(Base):
    def __init__(self, key, model_name, base_url):
        pass

    def similarity(self, query: str, texts: list):
        raise NotImplementedError("The api has not been implement")


# SILICONFLOW重排序模型实现类
class SILICONFLOWRerank(Base):
    def __init__(
            self, key, model_name, base_url="https://api.siliconflow.cn/v1/rerank"
    ):
        if not base_url:
            base_url = "https://api.siliconflow.cn/v1/rerank"
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        }

    def similarity(self, query: str, texts: list):
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
            "return_documents": False,
            "max_chunks_per_doc": 1024,
            "overlap_tokens": 80,
        }
        # 调用API进行重排序
        response = requests.post(
            self.base_url, json=payload, headers=self.headers
        ).json()
        rank = np.zeros(len(texts), dtype=float)
        if "results" not in response:
            return rank, 0

        for d in response["results"]:
            rank[d["index"]] = d["relevance_score"]
        return (
            rank,
            response["meta"]["tokens"]["input_tokens"] + response["meta"]["tokens"]["output_tokens"],
        )


# 百度文心一言重排序模型实现类
class BaiduYiyanRerank(Base):
    def __init__(self, key, model_name, base_url=None):
        from qianfan.resources import Reranker

        key = json.loads(key)
        ak = key.get("yiyan_ak", "")
        sk = key.get("yiyan_sk", "")
        self.client = Reranker(ak=ak, sk=sk)
        self.model_name = model_name

    def similarity(self, query: str, texts: list):
        # 调用API进行重排序
        res = self.client.do(
            model=self.model_name,
            query=query,
            documents=texts,
            top_n=len(texts),
        ).body
        rank = np.zeros(len(texts), dtype=float)
        for d in res["results"]:
            rank[d["index"]] = d["relevance_score"]
        return rank, self.total_token_count(res)


# Voyage重排序模型实现类
class VoyageRerank(Base):
    def __init__(self, key, model_name, base_url=None):
        import voyageai

        self.client = voyageai.Client(api_key=key)
        self.model_name = model_name

    def similarity(self, query: str, texts: list):
        rank = np.zeros(len(texts), dtype=float)
        if not texts:
            return rank, 0
        # 调用API进行重排序
        res = self.client.rerank(
            query=query, documents=texts, model=self.model_name, top_k=len(texts)
        )
        for r in res.results:
            rank[r.index] = r.relevance_score
        return rank, res.total_tokens


# 通义千问重排序模型实现类
class QWenRerank(Base):
    def __init__(self, key, model_name='gte-rerank', base_url=None, **kwargs):
        import dashscope
        self.api_key = key
        self.model_name = dashscope.TextReRank.Models.gte_rerank if model_name is None else model_name

    def similarity(self, query: str, texts: list):
        import dashscope
        from http import HTTPStatus
        # 调用API进行重排序
        resp = dashscope.TextReRank.call(
            api_key=self.api_key,
            model=self.model_name,
            query=query,
            documents=texts,
            top_n=len(texts),
            return_documents=False
        )
        rank = np.zeros(len(texts), dtype=float)
        if resp.status_code == HTTPStatus.OK:
            for r in resp.output.results:
                rank[r.index] = r.relevance_score
            return rank, resp.usage.total_tokens
        else:
            raise ValueError(f"Error calling QWenRerank model {self.model_name}: {resp.status_code} - {resp.text}")


# GPUStack重排序模型实现类
class GPUStackRerank(Base):
    def __init__(
            self, key, model_name, base_url
    ):
        if not base_url:
            raise ValueError("url cannot be None")

        self.model_name = model_name
        self.base_url = str(URL(base_url)/ "v1" / "rerank")
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        }

    def similarity(self, query: str, texts: list):
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }

        try:
            # 调用API进行重排序
            response = requests.post(
                self.base_url, json=payload, headers=self.headers
            )
            response.raise_for_status()
            response_json = response.json()

            rank = np.zeros(len(texts), dtype=float)
            if "results" not in response_json:
                return rank, 0

            token_count = 0
            for t in texts:
                token_count += num_tokens_from_string(t)

            for result in response_json["results"]:
                rank[result["index"]] = result["relevance_score"]

            return (
                rank,
                token_count,
            )

        except httpx.HTTPStatusError as e:
            raise ValueError(f"Error calling GPUStackRerank model {self.model_name}: {e.response.status_code} - {e.response.text}")

