#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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

# 导入所需的库
import logging  # 日志记录模块
from tika import parser  # Apache Tika解析器，用于处理文档
import re  # 正则表达式模块
from io import BytesIO  # 二进制数据处理

# 从自定义模块导入函数和类
from deepdoc.parser.utils import get_text  # 获取文本的工具函数
from rag.nlp import bullets_category, is_english,remove_contents_table, \
    hierarchical_merge, make_colon_as_title, naive_merge, random_choices, tokenize_table, \
    tokenize_chunks  # NLP处理相关功能
from rag.nlp import rag_tokenizer  # RAG分词器
from deepdoc.parser import PdfParser, DocxParser, PlainParser, HtmlParser  # 文档解析器


class Pdf(PdfParser):
    """
    PDF文档解析器类，继承自PdfParser
    用于处理PDF文档，执行OCR、布局分析、表格识别等任务
    """
    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None):
        """
        处理PDF文档的主函数
        
        参数:
            filename: PDF文件名
            binary: 二进制数据，如果提供则使用二进制数据而非文件
            from_page: 起始页码
            to_page: 结束页码
            zoomin: 缩放系数，用于处理图像
            callback: 回调函数，用于报告处理进度
        
        返回:
            包含提取文本和布局信息的元组列表，以及表格信息
        """
        from timeit import default_timer as timer
        start = timer()
        callback(msg="OCR started")
        # 处理图像
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback)
        callback(msg="OCR finished ({:.2f}s)".format(timer() - start))

        # 布局识别
        start = timer()
        self._layouts_rec(zoomin)
        callback(0.67, "Layout analysis ({:.2f}s)".format(timer() - start))
        logging.debug("layouts: {}".format(timer() - start))

        # 表格分析
        start = timer()
        self._table_transformer_job(zoomin)
        callback(0.68, "Table analysis ({:.2f}s)".format(timer() - start))

        # 文本提取和合并
        start = timer()
        self._text_merge()
        tbls = self._extract_table_figure(True, zoomin, True, True)
        self._naive_vertical_merge()
        self._filter_forpages()
        self._merge_with_same_bullet()
        callback(0.8, "Text extraction ({:.2f}s)".format(timer() - start))

        # 返回提取的文本和表格
        return [(b["text"] + self._line_tag(b, zoomin), b.get("layoutno", ""))
                for b in self.boxes], tbls


def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
    文档分块函数，支持多种文件格式的处理和文本分块
    
    参数:
        filename: 文件名
        binary: 二进制数据，可选
        from_page: 起始页码，默认为0
        to_page: 结束页码，默认为100000
        lang: 语言，默认为"Chinese"
        callback: 回调函数，用于报告进度
        **kwargs: 其他参数
    
    说明:
        支持的文件格式有docx, pdf, txt等
        对于PDF文件，由于文档可能很长且并非所有部分都有用，
        建议为每本书设置页码范围，以消除负面影响并节省计算时间。
    
    返回:
        处理后的文本块列表，适用于RAG（检索增强生成）
    """
    # 初始化文档信息字典
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    pdf_parser = None
    sections, tbls = [], []
    
    # 根据文件扩展名选择相应的解析器进行处理
    if re.search(r"\.docx$", filename, re.IGNORECASE):
        # 处理DOCX文件
        callback(0.1, "Start to parse.")
        doc_parser = DocxParser()
        # TODO: table of contents need to be removed
        sections, tbls = doc_parser(
            binary if binary else filename, from_page=from_page, to_page=to_page)
        # 移除目录内容，并检测是否为英文
        remove_contents_table(sections, eng=is_english(
            random_choices([t for t, _ in sections], k=200)))
        tbls = [((None, lns), None) for lns in tbls]
        callback(0.8, "Finish parsing.")

    elif re.search(r"\.pdf$", filename, re.IGNORECASE):
        # 处理PDF文件
        pdf_parser = Pdf()
        if kwargs.get("layout_recognize", "DeepDOC") == "Plain Text":
            pdf_parser = PlainParser()
        sections, tbls = pdf_parser(filename if not binary else binary,
                                    from_page=from_page, to_page=to_page, callback=callback)

    elif re.search(r"\.txt$", filename, re.IGNORECASE):
        # 处理TXT文件
        callback(0.1, "Start to parse.")
        txt = get_text(filename, binary)
        sections = txt.split("\n")
        sections = [(line, "") for line in sections if line]
        remove_contents_table(sections, eng=is_english(
            random_choices([t for t, _ in sections], k=200)))
        callback(0.8, "Finish parsing.")

    elif re.search(r"\.(htm|html)$", filename, re.IGNORECASE):
        # 处理HTML文件
        callback(0.1, "Start to parse.")
        sections = HtmlParser()(filename, binary)
        sections = [(line, "") for line in sections if line]
        remove_contents_table(sections, eng=is_english(
            random_choices([t for t, _ in sections], k=200)))
        callback(0.8, "Finish parsing.")

    elif re.search(r"\.doc$", filename, re.IGNORECASE):
        # 处理DOC文件，使用Tika解析器
        callback(0.1, "Start to parse.")
        binary = BytesIO(binary)
        doc_parsed = parser.from_buffer(binary)
        sections = doc_parsed['content'].split('\n')
        sections = [(line, "") for line in sections if line]
        remove_contents_table(sections, eng=is_english(
            random_choices([t for t, _ in sections], k=200)))
        callback(0.8, "Finish parsing.")

    else:
        # 不支持的文件格式
        raise NotImplementedError(
            "file type not supported yet(doc, docx, pdf, txt supported)")

    # 对段落进行后处理
    # 将冒号作为标题标记
    make_colon_as_title(sections)
    
    # 识别项目符号类别
    bull = bullets_category(
        [t for t in random_choices([t for t, _ in sections], k=100)])
    
    # 根据项目符号类别选择合并策略
    if bull >= 0:
        # 层次化合并
        chunks = ["\n".join(ck)
                  for ck in hierarchical_merge(bull, sections, 5)]
    else:
        # 普通合并
        sections = [s.split("@") for s, _ in sections]
        sections = [(pr[0], "@" + pr[1]) if len(pr) == 2 else (pr[0], '') for pr in sections ]
        chunks = naive_merge(
            sections, kwargs.get(
                "chunk_token_num", 256), kwargs.get(
                "delimer", "\n。；！？"))

    # 确定语言类型
    eng = lang.lower() == "english"

    # 对表格和文本块进行分词处理
    res = tokenize_table(tbls, doc, eng)
    res.extend(tokenize_chunks(chunks, doc, eng, pdf_parser))

    return res


# 主程序入口
if __name__ == "__main__":
    import sys

    # 定义一个空的回调函数
    def dummy(prog=None, msg=""):
        pass
        
    # 调用chunk函数处理命令行参数指定的文件
    chunk(sys.argv[1], from_page=1, to_page=10, callback=dummy)
