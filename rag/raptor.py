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
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED, wait
from threading import Lock
import umap  # 用于降维处理
import numpy as np
from sklearn.mixture import GaussianMixture  # 高斯混合模型，用于聚类

# 导入缓存相关工具函数
from graphrag.utils import get_llm_cache, get_embed_cache, set_embed_cache, set_llm_cache
from rag.utils import truncate  # 文本截断工具


class RecursiveAbstractiveProcessing4TreeOrganizedRetrieval:
    """
    递归抽象处理树状组织检索（RAPTOR）
    
    通过递归聚类和摘要生成，将大量文本组织成一个层次化的树状结构，
    每个节点是对其子节点内容的摘要，实现对大量文档的有效组织和检索。
    """
    def __init__(self, max_cluster, llm_model, embd_model, prompt, max_token=512, threshold=0.1):
        """
        初始化RAPTOR算法
        
        参数:
            max_cluster: 最大聚类数量
            llm_model: 大语言模型，用于生成摘要
            embd_model: 嵌入模型，用于文本向量化
            prompt: 摘要生成的提示模板
            max_token: 摘要的最大token数量
            threshold: 聚类概率阈值，用于确定文本所属的聚类
        """
        self._max_cluster = max_cluster
        self._llm_model = llm_model
        self._embd_model = embd_model
        self._threshold = threshold
        self._prompt = prompt
        self._max_token = max_token

    def _chat(self, system, history, gen_conf):
        """
        调用大语言模型生成摘要
        
        参数:
            system: 系统提示
            history: 对话历史
            gen_conf: 生成配置
            
        返回:
            生成的摘要文本
        """
        # 尝试从缓存获取响应
        response = get_llm_cache(self._llm_model.llm_name, system, history, gen_conf)
        if response:
            return response
        
        # 调用LLM生成摘要
        response = self._llm_model.chat(system, history, gen_conf)
        # 移除思考过程（<think>标签之间的内容）
        response = re.sub(r"<think>.*</think>", "", response, flags=re.DOTALL)
        # 检测错误
        if response.find("**ERROR**") >= 0:
            raise Exception(response)
        
        # 缓存响应
        set_llm_cache(self._llm_model.llm_name, system, response, history, gen_conf)
        return response

    def _embedding_encode(self, txt):
        """
        为文本生成嵌入向量
        
        参数:
            txt: 需要编码的文本
            
        返回:
            文本的嵌入向量
        """
        # 尝试从缓存获取向量
        response = get_embed_cache(self._embd_model.llm_name, txt)
        if response is not None:
            return response
        
        # 生成新的嵌入向量
        embds, _ = self._embd_model.encode([txt])
        # 检查向量有效性
        if len(embds) < 1 or len(embds[0]) < 1:
            raise Exception("Embedding error: ")
        embds = embds[0]
        
        # 缓存向量
        set_embed_cache(self._embd_model.llm_name, txt, embds)
        return embds

    def _get_optimal_clusters(self, embeddings: np.ndarray, random_state: int):
        """
        使用贝叶斯信息准则(BIC)确定最佳聚类数量
        
        参数:
            embeddings: 文本嵌入向量集合
            random_state: 随机种子，确保结果可复现
            
        返回:
            最优的聚类数量
        """
        # 设置最大聚类数（不超过数据点数量）
        max_clusters = min(self._max_cluster, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        
        # 计算不同聚类数的BIC值
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        
        # 选择BIC最小的聚类数
        optimal_clusters = n_clusters[np.argmin(bics)]
        return optimal_clusters

    def __call__(self, chunks, random_state, callback=None):
        """
        执行RAPTOR算法的主流程
        
        参数:
            chunks: 文本块列表，每个元素为(文本, 嵌入向量)对
            random_state: 随机种子
            callback: 回调函数，用于报告进度
            
        返回:
            处理后的文本块列表，包含原始文本和生成的摘要
        """
        # 初始化层次跟踪
        layers = [(0, len(chunks))]
        start, end = 0, len(chunks)
        
        # 特殊情况处理
        if len(chunks) <= 1:
            return
        
        # 过滤有效文本块
        chunks = [(s, a) for s, a in chunks if s and len(a) > 0]

        def summarize(ck_idx, lock):
            """
            为一组文本生成摘要
            
            参数:
                ck_idx: 需要摘要的文本块索引
                lock: 线程锁，确保并发安全
                
            返回:
                可能的异常，正常情况返回None
            """
            nonlocal chunks
            try:
                # 提取文本内容
                texts = [chunks[i][0] for i in ck_idx]
                # 计算每个文本可用的最大长度
                len_per_chunk = int((self._llm_model.max_length - self._max_token) / len(texts))
                # 拼接并截断文本
                cluster_content = "\n".join([truncate(t, max(1, len_per_chunk)) for t in texts])
                
                # 调用LLM生成摘要
                cnt = self._chat(
                    "You're a helpful assistant.",
                    [{"role": "user", "content": self._prompt.format(cluster_content=cluster_content)}],
                    {"temperature": 0.3, "max_tokens": self._max_token}
                )
                
                # 清理输出中的特殊标记
                cnt = re.sub(
                    "(······\n由于长度的原因，回答被截断了，要继续吗？|For the content length reason, it stopped, continue?)",
                    "",
                    cnt
                )
                
                logging.debug(f"SUM: {cnt}")
                # 为摘要生成嵌入向量
                embds, _ = self._embd_model.encode([cnt])
                
                # 线程安全地添加摘要到文本块列表
                with lock:
                    chunks.append((cnt, self._embedding_encode(cnt)))
            except Exception as e:
                logging.exception("summarize got exception")
                return e

        # 标签列表，用于跟踪文本块所属的聚类
        labels = []
        
        # 主循环，直到所有文本都被合并到一个摘要中
        while end - start > 1:
            # 提取当前层的嵌入向量
            embeddings = [embd for _, embd in chunks[start: end]]
            
            # 特殊情况：只有两个文本块时的处理
            if len(embeddings) == 2:
                summarize([start, start + 1], Lock())
                if callback:
                    callback(msg="Cluster one layer: {} -> {}".format(end - start, len(chunks) - end))
                labels.extend([0, 0])
                layers.append((end, len(chunks)))
                start = end
                end = len(chunks)
                continue

            # 计算UMAP降维的邻居数
            n_neighbors = int((len(embeddings) - 1) ** 0.8)
            
            # 使用UMAP进行降维，减少计算复杂度
            reduced_embeddings = umap.UMAP(
                n_neighbors=max(2, n_neighbors),
                n_components=min(12, len(embeddings) - 2),
                metric="cosine"
            ).fit_transform(embeddings)
            
            # 获取最佳聚类数
            n_clusters = self._get_optimal_clusters(reduced_embeddings, random_state)
            
            # 特殊情况：只有一个聚类时的处理
            if n_clusters == 1:
                lbls = [0 for _ in range(len(reduced_embeddings))]
            else:
                # 使用高斯混合模型进行聚类
                gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
                gm.fit(reduced_embeddings)
                
                # 获取每个点所属聚类的概率
                probs = gm.predict_proba(reduced_embeddings)
                
                # 根据概率阈值确定文本所属的聚类
                lbls = [np.where(prob > self._threshold)[0] for prob in probs]
                lbls = [lbl[0] if isinstance(lbl, np.ndarray) else lbl for lbl in lbls]
            
            # 并行处理各个聚类的摘要生成
            lock = Lock()
            with ThreadPoolExecutor(max_workers=12) as executor:
                threads = []
                
                # 为每个聚类提交摘要生成任务
                for c in range(n_clusters):
                    # 找出属于当前聚类的文本索引
                    ck_idx = [i + start for i in range(len(lbls)) if lbls[i] == c]
                    if not ck_idx:
                        continue
                    threads.append(executor.submit(summarize, ck_idx, lock))
                
                # 等待所有任务完成
                wait(threads, return_when=ALL_COMPLETED)
                
                # 检查是否有异常
                for th in threads:
                    if isinstance(th.result(), Exception):
                        raise th.result()
                
                logging.debug(str([t.result() for t in threads]))

            # 验证处理结果，确保生成了正确数量的摘要
            assert len(chunks) - end == n_clusters, "{} vs. {}".format(len(chunks) - end, n_clusters)
            
            # 更新标签和层次信息
            labels.extend(lbls)
            layers.append((end, len(chunks)))
            
            # 进度回调
            if callback:
                callback(msg="Cluster one layer: {} -> {}".format(end - start, len(chunks) - end))
            
            # 更新处理范围，指向下一层
            start = end
            end = len(chunks)

        # 返回处理结果
        return chunks

