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
from dataclasses import dataclass

from rag.settings import TAG_FLD, PAGERANK_FLD
from rag.utils import rmSpace, get_float
from rag.nlp import rag_tokenizer, query
import numpy as np
from rag.utils.doc_store_conn import DocStoreConnection, MatchDenseExpr, FusionExpr, OrderByExpr


def index_name(uid): return f"ragflow_{uid}"


class Dealer:
    """搜索处理器类
    提供文档搜索、重排序、引用插入等功能
    """
    def __init__(self, dataStore: DocStoreConnection):
        """初始化搜索处理器
        Args:
            dataStore: 文档存储连接器
        """
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        """搜索结果数据类
        Attributes:
            total: 结果总数
            ids: 文档ID列表
            query_vector: 查询向量
            field: 字段信息
            highlight: 高亮信息
            aggregation: 聚合信息
            keywords: 关键词列表
            group_docs: 分组文档列表
        """
        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None

    def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        """获取文本的向量表示
        Args:
            txt: 输入文本
            emb_mdl: 嵌入模型
            topk: 返回的top-k结果数
            similarity: 相似度阈值
        Returns:
            向量匹配表达式
        """
        qv, _ = emb_mdl.encode_queries(txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [get_float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

    def get_filters(self, req):
        """获取搜索过滤条件
        Args:
            req: 请求参数
        Returns:
            过滤条件字典
        """
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    def search(self, req, idx_names: str | list[str],
               kb_ids: list[str],
               emb_mdl=None,
               highlight=False,
               rank_feature: dict | None = None
               ):
        """执行搜索
        Args:
            req: 搜索请求参数
            idx_names: 索引名称
            kb_ids: 知识库ID列表
            emb_mdl: 嵌入模型
            highlight: 是否高亮
            rank_feature: 排序特征
        Returns:
            搜索结果
        """
        filters = self.get_filters(req)
        orderBy = OrderByExpr()

        # 分页参数处理
        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps

        # 获取需要返回的字段
        src = req.get("fields",
                      ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
                       "doc_id", "page_num_int", "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                       "question_kwd", "question_tks",
                       "available_int", "content_with_weight", PAGERANK_FLD, TAG_FLD])
        kwds = set([])

        # 处理查询
        qst = req.get("question", "")
        q_vec = []
        if not qst:
            if req.get("sort"):
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.getTotal(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            highlightFields = ["content_ltks", "title_tks"] if highlight else []
            matchText, keywords = self.qryr.question(qst, min_match=0.3)
            if emb_mdl is None:
                matchExprs = [matchText]
                res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            else:
                # 使用向量检索
                matchDense = self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
                q_vec = matchDense.embedding_data
                src.append(f"q_{len(q_vec)}_vec")

                # 融合检索结果
                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05, 0.95"})
                matchExprs = [matchText, matchDense, fusionExpr]

                res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))

                # 如果结果为空，降低匹配阈值重试
                if total == 0:
                    matchText, _ = self.qryr.question(qst, min_match=0.1)
                    filters.pop("doc_ids", None)
                    matchDense.extra_options["similarity"] = 0.17
                    res = self.dataStore.search(src, highlightFields, filters, [matchText, matchDense, fusionExpr],
                                                orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                    total = self.dataStore.getTotal(res)
                    logging.debug("Dealer.search 2 TOTAL: {}".format(total))

            # 提取关键词
            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)

        logging.debug(f"TOTAL: {total}")
        ids = self.dataStore.getChunkIds(res)
        keywords = list(kwds)
        highlight = self.dataStore.getHighlight(res, keywords, "content_with_weight")
        aggs = self.dataStore.getAggregation(res, "docnm_kwd")
        return self.SearchResult(
            total=total,
            ids=ids,
            query_vector=q_vec,
            aggregation=aggs,
            highlight=highlight,
            field=self.dataStore.getFields(res, src),
            keywords=keywords
        )

    @staticmethod
    def trans2floats(txt):
        """将文本转换为浮点数列表
        Args:
            txt: 输入文本
        Returns:
            浮点数列表
        """
        return [get_float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9):
        """在答案中插入引用
        Args:
            answer: 答案文本
            chunks: 文档块列表
            chunk_v: 文档块向量列表
            embd_mdl: 嵌入模型
            tkweight: 文本权重
            vtweight: 向量权重
        Returns:
            带引用的答案和引用集合
        """
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])
        # 处理代码块
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
        # 合并标点符号
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        # 过滤短文本
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        # 计算文本向量
        ans_v, _ = embd_mdl.encode(pieces_)
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0]*len(ans_v[0])
                logging.warning("The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        # 计算相似度并插入引用
        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
                      for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i],
                                                                chunk_v,
                                                                rag_tokenizer.tokenize(
                                                                    self.qryr.rmWWW(pieces_[i])).split(),
                                                                chunks_tks,
                                                                tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        # 生成带引用的答案
        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" ##{c}$$"
                seted.add(c)

        return res, seted

    def _rank_feature_scores(self, query_rfea, search_res):
        """计算排序特征分数
        Args:
            query_rfea: 查询排序特征
            search_res: 搜索结果
        Returns:
            排序特征分数列表
        """
        rank_fea = []
        for field, weight in query_rfea.items():
            if field not in search_res.field:
                continue
            scores = []
            for doc in search_res.field[field]:
                if doc is None:
                    scores.append(0)
                else:
                    scores.append(float(doc))
            rank_fea.append((scores, weight))
        return rank_fea

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: dict | None = None
               ):
        _, keywords = self.qryr.question(query)
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [get_float(v) for v in vector.split("\t")]
            ins_embd.append(vector)
        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: dict | None = None):
        """使用模型重排序搜索结果
        Args:
            rerank_mdl: 重排序模型
            sres: 搜索结果
            query: 查询文本
            tkweight: 文本权重
            vtweight: 向量权重
            cfield: 内容字段
            rank_feature: 排序特征
        Returns:
            重排序后的结果
        """
        if not sres.ids:
            return sres
        if rank_feature:
            rank_fea = self._rank_feature_scores(rank_feature, sres)
        else:
            rank_fea = []

        # 准备重排序数据
        pairs = []
        for doc in sres.field[cfield]:
            if doc is None:
                pairs.append("")
            else:
                pairs.append((query, doc))

        # 使用模型重排序
        scores = rerank_mdl.rerank(pairs)
        if rank_fea:
            for rfea, weight in rank_fea:
                scores = [s + w * r for s, r, w in zip(scores, rfea, [weight] * len(scores))]

        # 排序并更新结果
        idx = np.argsort(scores)[::-1]
        sres.ids = [sres.ids[i] for i in idx]
        for field in sres.field:
            sres.field[field] = [sres.field[field][i] for i in idx]
        if sres.highlight:
            sres.highlight = [sres.highlight[i] for i in idx]
        return sres

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        """计算混合相似度
        Args:
            ans_embd: 答案向量
            ins_embd: 实例向量
            ans: 答案文本
            inst: 实例文本
        Returns:
            相似度分数
        """
        return self.qryr.hybrid_similarity(ans_embd, ins_embd, ans, inst)

    def retrieval(self, question, embd_mdl, tenant_ids, kb_ids, page, page_size, similarity_threshold=0.2,
                  vector_similarity_weight=0.3, top=1024, doc_ids=None, aggs=True,
                  rerank_mdl=None, highlight=False,
                  rank_feature: dict | None = {PAGERANK_FLD: 10}):
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks

        RERANK_LIMIT = 64
        req = {"kb_ids": kb_ids, "doc_ids": doc_ids, "page": page, "size": RERANK_LIMIT,
               "question": question, "vector": True, "topk": top,
               "similarity": similarity_threshold,
               "available_int": 1}

        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        sres = self.search(req, [index_name(tid) for tid in tenant_ids],
                           kb_ids, embd_mdl, highlight, rank_feature=rank_feature)
        ranks["total"] = sres.total

        if rerank_mdl and sres.total > 0:
            sim, tsim, vsim = self.rerank_by_model(rerank_mdl,
                                                   sres, question, 1 - vector_similarity_weight,
                                                   vector_similarity_weight,
                                                   rank_feature=rank_feature)
        else:
            sim, tsim, vsim = self.rerank(
                sres, question, 1 - vector_similarity_weight, vector_similarity_weight,
                rank_feature=rank_feature)
        idx = np.argsort(sim * -1)[(page - 1) * page_size:page * page_size]


        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim
        for i in idx:
            if sim[i] < similarity_threshold:
                break
            if len(ranks["chunks"]) >= page_size:
                if aggs:
                    continue
                break
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")
            position_int = chunk.get("position_int", [])
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim[i],
                "vector_similarity": vsim[i],
                "term_similarity": tsim[i],
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
            }
            if highlight and sres.highlight:
                if id in sres.highlight:
                    d["highlight"] = rmSpace(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
            ranks["chunks"].append(d)
            if dnm not in ranks["doc_aggs"]:
                ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
            ranks["doc_aggs"][dnm]["count"] += 1
        ranks["doc_aggs"] = [{"doc_name": k,
                              "doc_id": v["doc_id"],
                              "count": v["count"]} for k,
                                                       v in sorted(ranks["doc_aggs"].items(),
                                                                   key=lambda x: x[1]["count"] * -1)]
        ranks["chunks"] = ranks["chunks"][:page_size]

        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        """执行SQL查询
        Args:
            sql: SQL语句
            fetch_size: 获取大小
            format: 返回格式
        Returns:
            查询结果
        """
        return self.dataStore.sql(sql, fetch_size, format)

    def chunk_list(self, doc_id: str, tenant_id: str,
                   kb_ids: list[str], max_count=1024,
                   offset=0,
                   fields=["docnm_kwd", "content_with_weight", "img_id"]):
        """获取文档块列表
        Args:
            doc_id: 文档ID
            tenant_id: 租户ID
            kb_ids: 知识库ID列表
            max_count: 最大数量
            offset: 偏移量
            fields: 返回字段
        Returns:
            文档块列表
        """
        return self.dataStore.chunk_list(doc_id, tenant_id, kb_ids, max_count, offset, fields)

    def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        """获取所有标签
        Args:
            tenant_id: 租户ID
            kb_ids: 知识库ID列表
            S: 返回数量
        Returns:
            标签列表
        """
        return self.dataStore.all_tags(tenant_id, kb_ids, S)

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        """获取部分标签
        Args:
            tenant_id: 租户ID
            kb_ids: 知识库ID列表
            S: 返回数量
        Returns:
            标签列表
        """
        return self.dataStore.all_tags_in_portion(tenant_id, kb_ids, S)

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []), keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a.replace(".", "_"): c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        return {a.replace(".", "_"): max(1, c) for a, c in tag_fea}
