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
import json
from copy import deepcopy
from functools import partial

import pandas as pd

from agent.component import component_class
from agent.component.base import ComponentBase


class Canvas:
    """
    dsl = {
        "components": {
            "begin": {
                "obj":{
                    "component_name": "Begin",
                    "params": {},
                },
                "downstream": ["answer_0"],
                "upstream": [],
            },
            "answer_0": {
                "obj": {
                    "component_name": "Answer",
                    "params": {}
                },
                "downstream": ["retrieval_0"],
                "upstream": ["begin", "generate_0"],
            },
            "retrieval_0": {
                "obj": {
                    "component_name": "Retrieval",
                    "params": {}
                },
                "downstream": ["generate_0"],
                "upstream": ["answer_0"],
            },
            "generate_0": {
                "obj": {
                    "component_name": "Generate",
                    "params": {}
                },
                "downstream": ["answer_0"],
                "upstream": ["retrieval_0"],
            }
        },
        "history": [],
        "messages": [],
        "reference": [],
        "path": [["begin"]],
        "answer": []
    }
    """

    def __init__(self, dsl: str, tenant_id=None):
        self.path = []
        self.history = []
        self.messages = []
        self.answer = []
        self.components = {}
        self.dsl = json.loads(dsl) if dsl else {
            "components": {
                "begin": {
                    "obj": {
                        "component_name": "Begin",
                        "params": {
                            "prologue": "Hi there!"
                        }
                    },
                    "downstream": [],
                    "upstream": [],
                    "parent_id": ""
                }
            },
            "history": [],
            "messages": [],
            "reference": [],
            "path": [],
            "answer": []
        }
        self._tenant_id = tenant_id
        self._embed_id = ""
        self.load()

    def load(self):
        self.components = self.dsl["components"]
        cpn_nms = set([])
        for k, cpn in self.components.items():
            cpn_nms.add(cpn["obj"]["component_name"])

        assert "Begin" in cpn_nms, "There have to be an 'Begin' component."
        assert "Answer" in cpn_nms, "There have to be an 'Answer' component."

        for k, cpn in self.components.items():
            cpn_nms.add(cpn["obj"]["component_name"])
            param = component_class(cpn["obj"]["component_name"] + "Param")()
            param.update(cpn["obj"]["params"])
            param.check()
            cpn["obj"] = component_class(cpn["obj"]["component_name"])(self, k, param)
            if cpn["obj"].component_name == "Categorize":
                for _, desc in param.category_description.items():
                    if desc["to"] not in cpn["downstream"]:
                        cpn["downstream"].append(desc["to"])

        self.path = self.dsl["path"]
        self.history = self.dsl["history"]
        self.messages = self.dsl["messages"]
        self.answer = self.dsl["answer"]
        self.reference = self.dsl["reference"]
        self._embed_id = self.dsl.get("embed_id", "")

    def __str__(self):
        self.dsl["path"] = self.path
        self.dsl["history"] = self.history
        self.dsl["messages"] = self.messages
        self.dsl["answer"] = self.answer
        self.dsl["reference"] = self.reference
        self.dsl["embed_id"] = self._embed_id
        dsl = {
            "components": {}
        }
        for k in self.dsl.keys():
            if k in ["components"]:
                continue
            dsl[k] = deepcopy(self.dsl[k])

        for k, cpn in self.components.items():
            if k not in dsl["components"]:
                dsl["components"][k] = {}
            for c in cpn.keys():
                if c == "obj":
                    dsl["components"][k][c] = json.loads(str(cpn["obj"]))
                    continue
                dsl["components"][k][c] = deepcopy(cpn[c])
        return json.dumps(dsl, ensure_ascii=False)

    def reset(self):
        self.path = []
        self.history = []
        self.messages = []
        self.answer = []
        self.reference = []
        for k, cpn in self.components.items():
            self.components[k]["obj"].reset()
        self._embed_id = ""

    def get_component_name(self, cid):
        for n in self.dsl["graph"]["nodes"]:
            if cid == n["id"]:
                return n["data"]["name"]
        return ""

    def run(self, **kwargs):
        if self.answer:
            cpn_id = self.answer[0]
            self.answer.pop(0)
            try:
                ans = self.components[cpn_id]["obj"].run(self.history, **kwargs)
            except Exception as e:
                ans = ComponentBase.be_output(str(e))
            self.path[-1].append(cpn_id)
            if kwargs.get("stream"):
                for an in ans():
                    yield an
            else:
                yield ans
            return

        if not self.path:
            self.components["begin"]["obj"].run(self.history, **kwargs)
            self.path.append(["begin"])

        self.path.append([])

        ran = -1
        waiting = []
        without_dependent_checking = []

        def prepare2run(cpns):
            nonlocal ran, ans
            for c in cpns:
                if self.path[-1] and c == self.path[-1][-1]:
                    continue
                cpn = self.components[c]["obj"]
                if cpn.component_name == "Answer":
                    self.answer.append(c)
                else:
                    logging.debug(f"Canvas.prepare2run: {c}")
                    if c not in without_dependent_checking:
                        cpids = cpn.get_dependent_components()
                        if any([cc not in self.path[-1] for cc in cpids]):
                            if c not in waiting:
                                waiting.append(c)
                            continue
                    yield "*'{}'* is running...🕞".format(self.get_component_name(c))

                    if cpn.component_name.lower() == "iteration":
                        st_cpn = cpn.get_start()
                        assert st_cpn, "Start component not found for Iteration."
                        if not st_cpn["obj"].end():
                            cpn = st_cpn["obj"]
                            c = cpn._id

                    try:
                        ans = cpn.run(self.history, **kwargs)
                    except Exception as e:
                        logging.exception(f"Canvas.run got exception: {e}")
                        self.path[-1].append(c)
                        ran += 1
                        raise e
                    self.path[-1].append(c)

            ran += 1

        downstream = self.components[self.path[-2][-1]]["downstream"]
        if not downstream and self.components[self.path[-2][-1]].get("parent_id"):
            cid = self.path[-2][-1]
            pid = self.components[cid]["parent_id"]
            o, _ = self.components[cid]["obj"].output(allow_partial=False)
            oo, _ = self.components[pid]["obj"].output(allow_partial=False)
            self.components[pid]["obj"].set_output(pd.concat([oo, o], ignore_index=True).dropna())
            downstream = [pid]

        for m in prepare2run(downstream):
            yield {"content": m, "running_status": True}

        while 0 <= ran < len(self.path[-1]):
            logging.debug(f"Canvas.run: {ran} {self.path}")
            cpn_id = self.path[-1][ran]
            cpn = self.get_component(cpn_id)
            if not any([cpn["downstream"], cpn.get("parent_id"), waiting]):
                break

            loop = self._find_loop()
            if loop:
                raise OverflowError(f"Too much loops: {loop}")

            if cpn["obj"].component_name.lower() in ["switch", "categorize", "relevant"]:
                switch_out = cpn["obj"].output()[1].iloc[0, 0]
                assert switch_out in self.components, \
                    "{}'s output: {} not valid.".format(cpn_id, switch_out)
                for m in prepare2run([switch_out]):
                    yield {"content": m, "running_status": True}
                continue

            downstream = cpn["downstream"]
            if not downstream and cpn.get("parent_id"):
                pid = cpn["parent_id"]
                _, o = cpn["obj"].output(allow_partial=False)
                _, oo = self.components[pid]["obj"].output(allow_partial=False)
                self.components[pid]["obj"].set_output(pd.concat([oo.dropna(axis=1), o.dropna(axis=1)], ignore_index=True).dropna())
                downstream = [pid]

            for m in prepare2run(downstream):
                yield {"content": m, "running_status": True}

            if ran >= len(self.path[-1]) and waiting:
                without_dependent_checking = waiting
                waiting = []
                for m in prepare2run(without_dependent_checking):
                    yield {"content": m, "running_status": True}
                without_dependent_checking = []
                ran -= 1

        if self.answer:
            cpn_id = self.answer[0]
            self.answer.pop(0)
            ans = self.components[cpn_id]["obj"].run(self.history, **kwargs)
            self.path[-1].append(cpn_id)
            if kwargs.get("stream"):
                assert isinstance(ans, partial)
                for an in ans():
                    yield an
            else:
                yield ans

        else:
            raise Exception("The dialog flow has no way to interact with you. Please add an 'Interact' component to the end of the flow.")

    def get_component(self, cpn_id):
        return self.components[cpn_id]

    def get_tenant_id(self):
        return self._tenant_id

    def get_history(self, window_size):
        convs = []
        for role, obj in self.history[window_size * -1:]:
            if isinstance(obj, list) and obj and all([isinstance(o, dict) for o in obj]):
                convs.append({"role": role, "content": '\n'.join([str(s.get("content", "")) for s in obj])})
            else:
                convs.append({"role": role, "content": str(obj)})
        return convs

    def add_user_input(self, question):
        self.history.append(("user", question))

    def set_embedding_model(self, embed_id):
        self._embed_id = embed_id

    def get_embedding_model(self):
        return self._embed_id

    def _find_loop(self, max_loops=6):
        path = self.path[-1][::-1]
        if len(path) < 2:
            return False

        for i in range(len(path)):
            if path[i].lower().find("answer") == 0 or path[i].lower().find("iterationitem") == 0:
                path = path[:i]
                break

        if len(path) < 2:
            return False

        for loc in range(2, len(path) // 2):
            pat = ",".join(path[0:loc])
            path_str = ",".join(path)
            if len(pat) >= len(path_str):
                return False
            loop = max_loops
            while path_str.find(pat) == 0 and loop >= 0:
                loop -= 1
                if len(pat)+1 >= len(path_str):
                    return False
                path_str = path_str[len(pat)+1:]
            if loop < 0:
                pat = " => ".join([p.split(":")[0] for p in path[0:loc]])
                return pat + " => " + pat

        return False

    def get_prologue(self):
        return self.components["begin"]["obj"]._param.prologue

    def set_global_param(self, **kwargs):
        for k, v in kwargs.items():
            for q in self.components["begin"]["obj"]._param.query:
                if k != q["key"]:
                    continue
                q["value"] = v

    def get_preset_param(self):
        return self.components["begin"]["obj"]._param.query

    def get_component_input_elements(self, cpnnm):
        return self.components[cpnnm]["obj"].get_input_elements()
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
import json
from abc import ABC
from copy import deepcopy
from functools import partial

import pandas as pd

from agent.component import component_class
from agent.component.base import ComponentBase


# Canvas类 - 工作流引擎的核心实现
class Canvas(ABC):
    """
    Canvas类用于管理和执行工作流
    
    DSL示例结构:
    dsl = {
        "components": {                # 组件配置
            "begin": {                 # 开始组件
                "obj":{
                    "component_name": "Begin",  # 组件类型
                    "params": {},              # 组件参数
                },
                "downstream": ["answer_0"],    # 下游连接的组件
                "upstream": [],                # 上游连接的组件
            },
            "answer_0": {
                "obj": {
                    "component_name": "Answer",
                    "params": {}
                },
                "downstream": ["retrieval_0"],
                "upstream": ["begin", "generate_0"],
            },
            "retrieval_0": {
                "obj": {
                    "component_name": "Retrieval",
                    "params": {}
                },
                "downstream": ["generate_0"],
                "upstream": ["answer_0"],
            },
            "generate_0": {
                "obj": {
                    "component_name": "Generate",
                    "params": {}
                },
                "downstream": ["answer_0"],
                "upstream": ["retrieval_0"],
            }
        },
        "history": [],    # 历史记录
        "messages": [],   # 消息列表
        "reference": [],  # 引用信息
        "path": [["begin"]],      # 执行路径
        "answer": []     # 答案列表
    }
    """

    def __init__(self, dsl: str, tenant_id=None):
        """
        初始化Canvas实例
        
        Args:
            dsl: DSL配置字符串
            tenant_id: 租户ID
        """
        self.path = []          # 执行路径
        self.history = []       # 历史记录
        self.messages = []      # 消息列表
        self.answer = []        # 答案列表
        self.components = {}    # 组件字典
        # 如果没有提供DSL，创建默认配置
        self.dsl = json.loads(dsl) if dsl else {
            "components": {
                "begin": {
                    "obj": {
                        "component_name": "Begin",
                        "params": {
                            "prologue": "Hi there!"  # 开场白
                        }
                    },
                    "downstream": [],
                    "upstream": [],
                    "parent_id": ""
                }
            },
            "history": [],
            "messages": [],
            "reference": [],
            "path": [],
            "answer": []
        }
        self._tenant_id = tenant_id
        self._embed_id = ""
        self.load()

    def load(self):
        """
        加载并验证DSL配置
        - 检查必需组件
        - 初始化组件参数
        - 验证组件配置
        """
        self.components = self.dsl["components"]
        cpn_nms = set([])
        # 收集所有组件名称
        for k, cpn in self.components.items():
            cpn_nms.add(cpn["obj"]["component_name"])

        # 验证必需组件
        assert "Begin" in cpn_nms, "必须包含'Begin'组件"
        assert "Answer" in cpn_nms, "必须包含'Answer'组件"

        # 初始化每个组件
        for k, cpn in self.components.items():
            cpn_nms.add(cpn["obj"]["component_name"])
            # 创建组件参数实例
            param = component_class(cpn["obj"]["component_name"] + "Param")()
            param.update(cpn["obj"]["params"])
            param.check()
            # 创建组件实例
            cpn["obj"] = component_class(cpn["obj"]["component_name"])(self, k, param)
            # 处理Categorize组件的特殊逻辑
            if cpn["obj"].component_name == "Categorize":
                for _, desc in param.category_description.items():
                    if desc["to"] not in cpn["downstream"]:
                        cpn["downstream"].append(desc["to"])

        # 加载其他配置
        self.path = self.dsl["path"]
        self.history = self.dsl["history"]
        self.messages = self.dsl["messages"]
        self.answer = self.dsl["answer"]
        self.reference = self.dsl["reference"]
        self._embed_id = self.dsl.get("embed_id", "")

    def __str__(self):
        self.dsl["path"] = self.path
        self.dsl["history"] = self.history
        self.dsl["messages"] = self.messages
        self.dsl["answer"] = self.answer
        self.dsl["reference"] = self.reference
        self.dsl["embed_id"] = self._embed_id
        dsl = {
            "components": {}
        }
        for k in self.dsl.keys():
            if k in ["components"]:
                continue
            dsl[k] = deepcopy(self.dsl[k])

        for k, cpn in self.components.items():
            if k not in dsl["components"]:
                dsl["components"][k] = {}
            for c in cpn.keys():
                if c == "obj":
                    dsl["components"][k][c] = json.loads(str(cpn["obj"]))
                    continue
                dsl["components"][k][c] = deepcopy(cpn[c])
        return json.dumps(dsl, ensure_ascii=False)

    def reset(self):
        self.path = []
        self.history = []
        self.messages = []
        self.answer = []
        self.reference = []
        for k, cpn in self.components.items():
            self.components[k]["obj"].reset()
        self._embed_id = ""

    def get_compnent_name(self, cid):
        for n in self.dsl["graph"]["nodes"]:
            if cid == n["id"]:
                return n["data"]["name"]
        return ""

    def run(self, **kwargs):
        """
        执行工作流
        
        Args:
            **kwargs: 运行参数，支持stream等选项
            
        Yields:
            工作流执行结果
        """
        # 处理答案组件
        if self.answer:
            cpn_id = self.answer[0]
            self.answer.pop(0)
            try:
                ans = self.components[cpn_id]["obj"].run(self.history, **kwargs)
            except Exception as e:
                ans = ComponentBase.be_output(str(e))
            self.path[-1].append(cpn_id)
            if kwargs.get("stream"):
                for an in ans():
                    yield an
            else:
                yield ans
            return

        # 初始化执行路径
        if not self.path:
            self.components["begin"]["obj"].run(self.history, **kwargs)
            self.path.append(["begin"])

        self.path.append([])

        # 执行工作流逻辑
        ran = -1
        waiting = []
        without_dependent_checking = []

        def prepare2run(cpns):
            # 声明使用外部变量ran和ans
            nonlocal ran, ans
            # 遍历所有组件
            for c in cpns:
                # 如果当前路径不为空且组件已经是最后一个执行的,则跳过
                if self.path[-1] and c == self.path[-1][-1]:
                    continue
                # 获取组件对象
                cpn = self.components[c]["obj"]
                # 如果是Answer组件,加入answer列表等待执行
                if cpn.component_name == "Answer":
                    self.answer.append(c)
                else:
                    # 记录调试日志
                    logging.debug(f"Canvas.prepare2run: {c}")
                    # 如果组件不在无需检查依赖的列表中
                    if c not in without_dependent_checking:
                        # 获取该组件依赖的其他组件ID
                        cpids = cpn.get_dependent_components()
                        # 如果有任何依赖组件未执行
                        if any([cc not in self.path[-1] for cc in cpids]):
                            # 将组件加入等待列表并跳过
                            if c not in waiting:
                                waiting.append(c)
                            continue
                    # 输出组件运行状态
                    yield "*'{}'* is running...🕞".format(self.get_compnent_name(c))

                    # 如果是迭代组件
                    if cpn.component_name.lower() == "iteration":
                        # 获取迭代的起始组件
                        st_cpn = cpn.get_start()
                        # 确保存在起始组件
                        assert st_cpn, "Start component not found for Iteration."
                        # 如果起始组件未结束,则执行起始组件
                        if not st_cpn["obj"].end():
                            cpn = st_cpn["obj"]
                            c = cpn._id

                    # 执行组件
                    try:
                        ans = cpn.run(self.history, **kwargs)
                    except Exception as e:
                        # 记录异常日志
                        logging.exception(f"Canvas.run got exception: {e}")
                        # 将组件加入执行路径
                        self.path[-1].append(c)
                        ran += 1
                        raise e
                    # 将成功执行的组件加入执行路径
                    self.path[-1].append(c)

            # 完成一轮执行,计数器加1
            ran += 1

        # 获取上一个执行路径中最后一个组件的下游组件列表
        downstream = self.components[self.path[-2][-1]]["downstream"]
        
        # 如果没有下游组件且该组件有父组件
        if not downstream and self.components[self.path[-2][-1]].get("parent_id"):
            # 获取当前组件ID
            cid = self.path[-2][-1]
            # 获取父组件ID
            pid = self.components[cid]["parent_id"]
            # 获取当前组件的输出(不允许部分输出)
            o, _ = self.components[cid]["obj"].output(allow_partial=False)
            # 获取父组件的输出(不允许部分输出) 
            oo, _ = self.components[pid]["obj"].output(allow_partial=False)
            # 将当前组件和父组件的输出合并,并设置为父组件的新输出
            self.components[pid]["obj"].set(pd.concat([oo, o], ignore_index=True))
            # 将父组件ID设为下游组件
            downstream = [pid]

        # 遍历prepare2run生成器,准备执行下游组件
        for m in prepare2run(downstream):
            # 生成包含运行状态的消息字典
            yield {"content": m, "running_status": True}

        # 当运行计数器在有效范围内时继续执行
        while 0 <= ran < len(self.path[-1]):
            # 记录调试日志,显示当前运行状态
            logging.debug(f"Canvas.run: {ran} {self.path}")
            # 获取当前执行路径中的组件ID
            cpn_id = self.path[-1][ran]
            # 获取该组件的详细信息
            cpn = self.get_component(cpn_id)
            # 如果组件没有下游组件、父组件且没有等待执行的组件,则退出循环
            if not any([cpn["downstream"], cpn.get("parent_id"), waiting]):
                break

            # 检查是否存在循环
            loop = self._find_loop()
            # 如果检测到循环,抛出溢出错误
            if loop:
                raise OverflowError(f"Too much loops: {loop}")

            # 如果是分支类组件(switch/categorize/relevant)
            if cpn["obj"].component_name.lower() in ["switch", "categorize", "relevant"]:
                # 获取分支组件的输出结果(第一行第一列)
                switch_out = cpn["obj"].output()[1].iloc[0, 0]
                # 确保输出的组件ID存在于组件列表中
                assert switch_out in self.components, \
                    "{}'s output: {} not valid.".format(cpn_id, switch_out)
                # 准备执行输出指向的组件
                for m in prepare2run([switch_out]):
                    yield {"content": m, "running_status": True}
                continue

            # 获取当前组件的下游组件列表
            downstream = cpn["downstream"]
            # 如果没有下游组件但有父组件
            if not downstream and cpn.get("parent_id"):
                # 获取父组件ID
                pid = cpn["parent_id"]
                # 获取当前组件的输出(不允许部分输出)
                _, o = cpn["obj"].output(allow_partial=False)
                # 获取父组件的输出(不允许部分输出)
                _, oo = self.components[pid]["obj"].output(allow_partial=False)
                # 合并当前组件和父组件的输出,并设置为父组件的新输出
                self.components[pid]["obj"].set_output(pd.concat([oo.dropna(axis=1), o.dropna(axis=1)], ignore_index=True))
                # 将父组件ID设为下游组件
                downstream = [pid]

            # 准备执行下游组件
            for m in prepare2run(downstream):
                yield {"content": m, "running_status": True}

            # 如果运行计数超出路径长度且还有等待执行的组件
            if ran >= len(self.path[-1]) and waiting:
                # 将等待列表赋值给无依赖检查列表
                without_dependent_checking = waiting
                # 清空等待列表
                waiting = []
                # 准备执行无依赖检查列表中的组件
                for m in prepare2run(without_dependent_checking):
                    yield {"content": m, "running_status": True}
                # 清空无依赖检查列表
                without_dependent_checking = []
                # 运行计数减1
                ran -= 1

        # 如果存在待回答的组件
        if self.answer:
            # 获取第一个待回答组件的ID
            cpn_id = self.answer[0]
            # 从待回答列表中移除该组件
            self.answer.pop(0)
            # 运行该组件,传入历史记录和其他参数
            ans = self.components[cpn_id]["obj"].run(self.history, **kwargs)
            # 将该组件ID添加到当前执行路径
            self.path[-1].append(cpn_id)
            # 如果开启了流式输出
            if kwargs.get("stream"):
                # 确保返回值是partial对象(用于延迟执行)
                assert isinstance(ans, partial)
                # 迭代生成流式输出
                for an in ans():
                    yield an
            # 如果是普通输出
            else:
                # 直接返回结果
                yield ans

        # 如果没有待回答的组件
        else:
            # 抛出异常,提示需要添加交互组件
            raise Exception("The dialog flow has no way to interact with you. Please add an 'Interact' component to the end of the flow.")

    def get_component(self, cpn_id):
        return self.components[cpn_id]

    def get_tenant_id(self):
        return self._tenant_id

    def get_history(self, window_size):
        convs = []
        for role, obj in self.history[window_size * -1:]:
            if isinstance(obj, list) and obj and all([isinstance(o, dict) for o in obj]):
                convs.append({"role": role, "content": '\n'.join([str(s.get("content", "")) for s in obj])})
            else:
                convs.append({"role": role, "content": str(obj)})
        return convs

    def add_user_input(self, question):
        self.history.append(("user", question))

    def set_embedding_model(self, embed_id):
        self._embed_id = embed_id

    def get_embedding_model(self):
        return self._embed_id

    def _find_loop(self, max_loops=6):
        """
        检测工作流中的循环
        
        Args:
            max_loops: 最大允许的循环次数
            
        Returns:
            False或循环路径描述
        """
        # 获取当前执行路径并反转顺序
        path = self.path[-1][::-1]
        # 如果路径长度小于2,无法形成循环
        if len(path) < 2:
            return False

        # 遍历路径,找到Answer或IterationItem组件的位置
        for i in range(len(path)):
            # 如果找到Answer或IterationItem组件
            if path[i].lower().find("answer") == 0 or path[i].lower().find("iterationitem") == 0:
                # 截取该组件之前的路径
                path = path[:i]
                break

        # 截取后的路径长度小于2,无法形成循环
        if len(path) < 2:
            return False

        # 遍历可能的循环长度(从2到路径长度的一半)
        for loc in range(2, len(path) // 2):
            # 构建可能的循环模式
            pat = ",".join(path[0:loc])
            # 构建完整路径字符串
            path_str = ",".join(path)
            # 如果模式长度大于等于路径长度,不可能形成循环
            if len(pat) >= len(path_str):
                return False
            # 初始化循环计数器
            loop = max_loops
            # 检查路径是否重复出现该模式
            while path_str.find(pat) == 0 and loop >= 0:
                loop -= 1
                # 如果剩余路径太短,不可能继续循环
                if len(pat)+1 >= len(path_str):
                    return False
                # 移除已匹配的模式部分
                path_str = path_str[len(pat)+1:]
            # 如果超过最大循环次数
            if loop < 0:
                # 构建循环路径描述(只保留组件名,不含ID)
                pat = " => ".join([p.split(":")[0] for p in path[0:loc]])
                # 返回循环路径描述
                return pat + " => " + pat

        # 未检测到循环
        return False

    def get_prologue(self):
        return self.components["begin"]["obj"]._param.prologue

    def set_global_param(self, **kwargs):
        for k, v in kwargs.items():
            for q in self.components["begin"]["obj"]._param.query:
                if k != q["key"]:
                    continue
                q["value"] = v

    def get_preset_param(self):
        return self.components["begin"]["obj"]._param.query

    def get_component_input_elements(self, cpnnm):
        return self.components[cpnnm]["obj"].get_input_elements()