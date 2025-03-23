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

# 导入必要的库和模块
import json
import os
import sys
import time
import argparse
from collections import defaultdict

from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from api.db.services.knowledgebase_service import KnowledgebaseService
from api import settings
from api.utils import get_uuid
from rag.nlp import tokenize, search
from ranx import evaluate
from ranx import Qrels, Run
import pandas as pd
from tqdm import tqdm

# 全局变量，用于限制处理的文档数量
global max_docs
max_docs = sys.maxsize


class Benchmark:
    """
    基准测试类，用于评估RAGFlow系统在标准数据集上的检索性能
    支持MS MARCO v1.1、TriviaQA和MIRACL等多个数据集的评估
    """
    def __init__(self, kb_id):
        """
        初始化基准测试类
        
        参数:
            kb_id: 知识库ID，用于加载相关配置信息
        """
        self.kb_id = kb_id
        # 获取知识库对象
        e, self.kb = KnowledgebaseService.get_by_id(kb_id)
        # 加载知识库的相似度阈值和向量相似度权重配置
        self.similarity_threshold = self.kb.similarity_threshold
        self.vector_similarity_weight = self.kb.vector_similarity_weight
        # 初始化嵌入模型
        self.embd_mdl = LLMBundle(self.kb.tenant_id, LLMType.EMBEDDING, llm_name=self.kb.embd_id, lang=self.kb.language)
        # 租户ID和索引名初始化为空，将在运行时设置
        self.tenant_id = ''
        self.index_name = ''
        self.initialized_index = False

    def _get_retrieval(self, qrels):
        """
        执行检索操作，获取每个查询的检索结果
        
        参数:
            qrels: 查询相关性字典，包含查询和相关文档的映射关系
            
        返回:
            run: 检索结果字典，包含查询和检索到的文档及其相似度分数
        """
        # 等待ES和Infinity索引准备完毕
        time.sleep(20)
        run = defaultdict(dict)
        query_list = list(qrels.keys())
        for query in query_list:
            # 对每个查询执行检索操作
            ranks = settings.retrievaler.retrieval(query, self.embd_mdl, self.tenant_id, [self.kb.id], 1, 30,
                                            0.0, self.vector_similarity_weight)
            if len(ranks["chunks"]) == 0:
                print(f"deleted query: {query}")
                del qrels[query]
                continue
            # 保存检索结果，移除向量以节省空间
            for c in ranks["chunks"]:
                c.pop("vector", None)
                run[query][c["chunk_id"]] = c["similarity"]
        return run

    def embedding(self, docs):
        """
        为文档生成嵌入向量
        
        参数:
            docs: 待处理的文档列表
            
        返回:
            docs: 添加嵌入向量后的文档列表
            vector_size: 嵌入向量的维度
        """
        # 提取文档内容
        texts = [d["content_with_weight"] for d in docs]
        # 生成嵌入向量
        embeddings, _ = self.embd_mdl.encode(texts)
        assert len(docs) == len(embeddings)
        vector_size = 0
        # 将嵌入向量添加到文档中
        for i, d in enumerate(docs):
            v = embeddings[i]
            vector_size = len(v)
            # 将向量存储为字典的值,键名为"q_向量维度_vec"格式,如"q_768_vec"表示768维向量
            d["q_%d_vec" % len(v)] = v
        return docs, vector_size

    def init_index(self, vector_size: int):
        """
        初始化向量索引
        
        参数:
            vector_size: 向量维度，用于创建索引
        """
        if self.initialized_index:
            return
        # 如果索引已存在，则先删除
        if settings.docStoreConn.indexExist(self.index_name, self.kb_id):
            settings.docStoreConn.deleteIdx(self.index_name, self.kb_id)
        # 创建新索引
        settings.docStoreConn.createIdx(self.index_name, self.kb_id, vector_size)
        self.initialized_index = True

    def ms_marco_index(self, file_path, index_name):
        """
        处理MS MARCO数据集并构建索引
        
        参数:
            file_path: MS MARCO数据集路径
            index_name: 索引名称
            
        返回:
            qrels: 查询相关性字典
            texts: 文档ID到文本内容的映射字典
        """
        qrels = defaultdict(dict)  # 存储查询与文档的相关性信息
        texts = defaultdict(dict)  # 存储文档ID与文本内容的映射
        docs_count = 0  # 已处理的文档计数
        docs = []  # 待处理的文档缓冲区
        filelist = sorted(os.listdir(file_path))

        for fn in filelist:
            if docs_count >= max_docs:
                break
            if not fn.endswith(".parquet"):
                continue
            # 读取parquet文件
            data = pd.read_parquet(os.path.join(file_path, fn))
            for i in tqdm(range(len(data)), colour="green", desc="Tokenizing:" + fn):
                if docs_count >= max_docs:
                    break
                query = data.iloc[i]['query']  # 获取查询文本
                # 处理每个查询对应的文章段落
                for rel, text in zip(data.iloc[i]['passages']['is_selected'], data.iloc[i]['passages']['passage_text']):
                    d = {
                        "id": get_uuid(),  # 生成唯一ID
                        "kb_id": self.kb.id,
                        "docnm_kwd": "xxxxx",
                        "doc_id": "ksksks"
                    }
                    # 对文本进行分词处理
                    tokenize(d, text, "english")
                    docs.append(d)
                    texts[d["id"]] = text
                    qrels[query][d["id"]] = int(rel)  # 保存查询-文档相关性
                # 当积累了足够多的文档时，批量生成嵌入并索引
                if len(docs) >= 32:
                    docs_count += len(docs)
                    docs, vector_size = self.embedding(docs)
                    self.init_index(vector_size)
                    settings.docStoreConn.insert(docs, self.index_name, self.kb_id)
                    docs = []

        # 处理剩余的文档
        if docs:
            docs, vector_size = self.embedding(docs)
            self.init_index(vector_size)
            settings.docStoreConn.insert(docs, self.index_name, self.kb_id)
        return qrels, texts

    def trivia_qa_index(self, file_path, index_name):
        """
        处理TriviaQA数据集并构建索引
        
        参数:
            file_path: TriviaQA数据集路径
            index_name: 索引名称
            
        返回:
            qrels: 查询相关性字典
            texts: 文档ID到文本内容的映射字典
        """
        qrels = defaultdict(dict)
        texts = defaultdict(dict)
        docs_count = 0
        docs = []
        filelist = sorted(os.listdir(file_path))
        for fn in filelist:
            if docs_count >= max_docs:
                break
            if not fn.endswith(".parquet"):
                continue
            # 读取parquet文件
            data = pd.read_parquet(os.path.join(file_path, fn))
            for i in tqdm(range(len(data)), colour="green", desc="Indexing:" + fn):
                if docs_count >= max_docs:
                    break
                query = data.iloc[i]['question']  # TriviaQA中查询是问题
                # 处理每个问题对应的搜索结果
                for rel, text in zip(data.iloc[i]["search_results"]['rank'],
                                     data.iloc[i]["search_results"]['search_context']):
                    d = {
                        "id": get_uuid(),
                        "kb_id": self.kb.id,
                        "docnm_kwd": "xxxxx",
                        "doc_id": "ksksks"
                    }
                    tokenize(d, text, "english")
                    docs.append(d)
                    texts[d["id"]] = text
                    qrels[query][d["id"]] = int(rel)
                # 批量处理文档
                if len(docs) >= 32:
                    docs_count += len(docs)
                    docs, vector_size = self.embedding(docs)
                    self.init_index(vector_size)
                    settings.docStoreConn.insert(docs,self.index_name)
                    docs = []

        # 处理剩余文档
        docs, vector_size = self.embedding(docs)
        self.init_index(vector_size)
        settings.docStoreConn.insert(docs, self.index_name)
        return qrels, texts

    def miracl_index(self, file_path, corpus_path, index_name):
        """
        处理MIRACL数据集并构建索引，MIRACL是多语言检索数据集
        
        参数:
            file_path: MIRACL数据集主路径
            corpus_path: MIRACL语料库路径
            index_name: 索引名称
            
        返回:
            qrels: 查询相关性字典
            texts: 文档ID到文本内容的映射字典
        """
        # 首先加载语料库数据
        corpus_total = {}
        for corpus_file in os.listdir(corpus_path):
            tmp_data = pd.read_json(os.path.join(corpus_path, corpus_file), lines=True)
            for index, i in tmp_data.iterrows():
                corpus_total[i['docid']] = i['text']

        # 加载主题/查询数据
        topics_total = {}
        for topics_file in os.listdir(os.path.join(file_path, 'topics')):
            if 'test' in topics_file:  # 跳过测试文件
                continue
            tmp_data = pd.read_csv(os.path.join(file_path, 'topics', topics_file), sep='\t', names=['qid', 'query'])
            for index, i in tmp_data.iterrows():
                topics_total[i['qid']] = i['query']

        qrels = defaultdict(dict)
        texts = defaultdict(dict)
        docs_count = 0
        docs = []
        # 处理相关性评估文件
        for qrels_file in os.listdir(os.path.join(file_path, 'qrels')):
            if 'test' in qrels_file:
                continue
            if docs_count >= max_docs:
                break

            tmp_data = pd.read_csv(os.path.join(file_path, 'qrels', qrels_file), sep='\t',
                                   names=['qid', 'Q0', 'docid', 'relevance'])
            for i in tqdm(range(len(tmp_data)), colour="green", desc="Indexing:" + qrels_file):
                if docs_count >= max_docs:
                    break
                # 从主题中获取查询文本
                query = topics_total[tmp_data.iloc[i]['qid']]
                # 从语料库中获取文档文本
                text = corpus_total[tmp_data.iloc[i]['docid']]
                rel = tmp_data.iloc[i]['relevance']
                d = {
                    "id": get_uuid(),
                    "kb_id": self.kb.id,
                    "docnm_kwd": "xxxxx",
                    "doc_id": "ksksks"
                }
                tokenize(d, text, 'english')
                docs.append(d)
                texts[d["id"]] = text
                qrels[query][d["id"]] = int(rel)
                # 批量处理文档
                if len(docs) >= 32:
                    docs_count += len(docs)
                    docs, vector_size = self.embedding(docs)
                    self.init_index(vector_size)
                    settings.docStoreConn.insert(docs, self.index_name)
                    docs = []

        # 处理剩余文档
        docs, vector_size = self.embedding(docs)
        self.init_index(vector_size)
        settings.docStoreConn.insert(docs, self.index_name)
        return qrels, texts

    def save_results(self, qrels, run, texts, dataset, file_path):
        """
        保存评估结果到文件
        
        参数:
            qrels: 查询相关性字典
            run: 检索结果字典
            texts: 文档ID到文本内容的映射
            dataset: 数据集名称，用于文件命名
            file_path: 结果保存路径
        """
        keep_result = []
        run_keys = list(run.keys())
        # 为每个查询计算NDCG@10评分
        for run_i in tqdm(range(len(run_keys)), desc="Calculating ndcg@10 for single query"):
            key = run_keys[run_i]
            keep_result.append({'query': key, 'qrel': qrels[key], 'run': run[key],
                                'ndcg@10': evaluate({key: qrels[key]}, {key: run[key]}, "ndcg@10")})
        # 按NDCG@10评分排序
        keep_result = sorted(keep_result, key=lambda kk: kk['ndcg@10'])
        
        # 将结果保存为Markdown格式
        with open(os.path.join(file_path, dataset + 'result.md'), 'w', encoding='utf-8') as f:
            f.write('## Score For Every Query\n')
            for keep_result_i in keep_result:
                f.write('### query: ' + keep_result_i['query'] + ' ndcg@10:' + str(keep_result_i['ndcg@10']) + '\n')
                scores = [[i[0], i[1]] for i in keep_result_i['run'].items()]
                scores = sorted(scores, key=lambda kk: kk[1])
                # 显示前10个检索结果
                for score in scores[:10]:
                    f.write('- text: ' + str(texts[score[0]]) + '\t qrel: ' + str(score[1]) + '\n')
        
        # 保存详细结果为JSON格式
        json.dump(qrels, open(os.path.join(file_path, dataset + '.qrels.json'), "w+", encoding='utf-8'), indent=2)
        json.dump(run, open(os.path.join(file_path, dataset + '.run.json'), "w+", encoding='utf-8'), indent=2)
        print(os.path.join(file_path, dataset + '_result.md'), 'Saved!')

    def __call__(self, dataset, file_path, miracl_corpus=''):
        """
        执行基准测试的主函数，根据数据集类型选择相应的处理流程
        
        参数:
            dataset: 数据集名称，支持"ms_marco_v1.1"、"trivia_qa"和"miracl"
            file_path: 数据集路径
            miracl_corpus: MIRACL语料库路径，仅当dataset为"miracl"时需要
        """
        if dataset == "ms_marco_v1.1":
            # 处理MS MARCO数据集
            self.tenant_id = "benchmark_ms_marco_v11"
            self.index_name = search.index_name(self.tenant_id)
            qrels, texts = self.ms_marco_index(file_path, "benchmark_ms_marco_v1.1")
            run = self._get_retrieval(qrels)
            # 评估多个指标：NDCG@10、MAP@5、MRR@10
            print(dataset, evaluate(Qrels(qrels), Run(run), ["ndcg@10", "map@5", "mrr@10"]))
            self.save_results(qrels, run, texts, dataset, file_path)
        if dataset == "trivia_qa":
            # 处理TriviaQA数据集
            self.tenant_id = "benchmark_trivia_qa"
            self.index_name = search.index_name(self.tenant_id)
            qrels, texts = self.trivia_qa_index(file_path, "benchmark_trivia_qa")
            run = self._get_retrieval(qrels)
            print(dataset, evaluate(Qrels(qrels), Run(run), ["ndcg@10", "map@5", "mrr@10"]))
            self.save_results(qrels, run, texts, dataset, file_path)
        if dataset == "miracl":
            # 处理MIRACL多语言数据集，支持18种语言
            for lang in ['ar', 'bn', 'de', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th',
                         'yo', 'zh']:
                # 检查必要的目录是否存在
                if not os.path.isdir(os.path.join(file_path, 'miracl-v1.0-' + lang)):
                    print('Directory: ' + os.path.join(file_path, 'miracl-v1.0-' + lang) + ' not found!')
                    continue
                if not os.path.isdir(os.path.join(file_path, 'miracl-v1.0-' + lang, 'qrels')):
                    print('Directory: ' + os.path.join(file_path, 'miracl-v1.0-' + lang, 'qrels') + 'not found!')
                    continue
                if not os.path.isdir(os.path.join(file_path, 'miracl-v1.0-' + lang, 'topics')):
                    print('Directory: ' + os.path.join(file_path, 'miracl-v1.0-' + lang, 'topics') + 'not found!')
                    continue
                if not os.path.isdir(os.path.join(miracl_corpus, 'miracl-corpus-v1.0-' + lang)):
                    print('Directory: ' + os.path.join(miracl_corpus, 'miracl-corpus-v1.0-' + lang) + ' not found!')
                    continue
                
                # 为每种语言设置租户ID和索引名
                self.tenant_id = "benchmark_miracl_" + lang
                self.index_name = search.index_name(self.tenant_id)
                self.initialized_index = False
                qrels, texts = self.miracl_index(os.path.join(file_path, 'miracl-v1.0-' + lang),
                                                 os.path.join(miracl_corpus, 'miracl-corpus-v1.0-' + lang),
                                                 "benchmark_miracl_" + lang)
                run = self._get_retrieval(qrels)
                print(dataset, evaluate(Qrels(qrels), Run(run), ["ndcg@10", "map@5", "mrr@10"]))
                self.save_results(qrels, run, texts, dataset, file_path)


# 主程序入口
if __name__ == '__main__':
    print('*****************RAGFlow Benchmark*****************')
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(usage="benchmark.py <max_docs> <kb_id> <dataset> <dataset_path> [<miracl_corpus_path>])", description='RAGFlow Benchmark')
    parser.add_argument('max_docs', metavar='max_docs', type=int, help='max docs to evaluate')
    parser.add_argument('kb_id', metavar='kb_id', help='knowledgebase id')
    parser.add_argument('dataset', metavar='dataset', help='dataset name, shall be one of ms_marco_v1.1(https://huggingface.co/datasets/microsoft/ms_marco), trivia_qa(https://huggingface.co/datasets/mandarjoshi/trivia_qa>), miracl(https://huggingface.co/datasets/miracl/miracl')
    parser.add_argument('dataset_path', metavar='dataset_path', help='dataset path')
    parser.add_argument('miracl_corpus_path', metavar='miracl_corpus_path', nargs='?', default="", help='miracl corpus path. Only needed when dataset is miracl')

    # 解析命令行参数
    args = parser.parse_args()
    max_docs = args.max_docs  # 设置最大文档数量
    kb_id = args.kb_id  # 知识库ID
    ex = Benchmark(kb_id)  # 创建基准测试实例

    dataset = args.dataset  # 数据集名称
    dataset_path = args.dataset_path  # 数据集路径

    # 根据数据集类型执行相应的处理
    if dataset == "ms_marco_v1.1" or dataset == "trivia_qa":
        ex(dataset, dataset_path)
    elif dataset == "miracl":
        if len(args) < 5:
            print('Please input the correct parameters!')
            exit(1)
        miracl_corpus_path = args[4]
        ex(dataset, dataset_path, miracl_corpus=args.miracl_corpus_path)
    else:
        print("Dataset: ", dataset, "not supported!")
