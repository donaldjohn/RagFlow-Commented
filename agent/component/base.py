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
from abc import ABC
import builtins
import json
import os
import logging
from functools import partial
from typing import Tuple, Union

import pandas as pd

from agent import settings

_FEEDED_DEPRECATED_PARAMS = "_feeded_deprecated_params"
_DEPRECATED_PARAMS = "_deprecated_params"
_USER_FEEDED_PARAMS = "_user_feeded_params"
_IS_RAW_CONF = "_is_raw_conf"


class ComponentParamBase(ABC):
    """组件参数的基础类，用于管理和验证组件参数"""
    
    def __init__(self):
        """初始化基础参数"""
        self.output_var_name = "output"  # 输出变量名
        self.message_history_window_size = 22  # 消息历史窗口大小
        self.query = []  # 查询参数列表
        self.inputs = []  # 输入参数列表
        self.debug_inputs = []  # 调试输入列表

    def set_name(self, name: str):
        """设置参数名称"""
        self._name = name
        return self

    def check(self):
        """检查参数有效性，需要子类实现"""
        raise NotImplementedError("Parameter Object should be checked.")

    @classmethod
    def _get_or_init_deprecated_params_set(cls):
        """获取或初始化类级别的已弃用参数集合
        用于记录整个类范围内所有标记为弃用的参数名称
        """
        if not hasattr(cls, _DEPRECATED_PARAMS):
            setattr(cls, _DEPRECATED_PARAMS, set())
        return getattr(cls, _DEPRECATED_PARAMS)

    def _get_or_init_feeded_deprecated_params_set(self, conf=None):
        """获取或初始化实例级别的已配置弃用参数集合
        用于记录当前实例中用户实际配置过的已弃用参数
        Args:
            conf: 配置信息，用于从现有配置初始化集合
        """
        if not hasattr(self, _FEEDED_DEPRECATED_PARAMS):
            if conf is None:
                setattr(self, _FEEDED_DEPRECATED_PARAMS, set())
            else:
                setattr(
                    self,
                    _FEEDED_DEPRECATED_PARAMS,
                    set(conf[_FEEDED_DEPRECATED_PARAMS]),
                )
        return getattr(self, _FEEDED_DEPRECATED_PARAMS)

    def _get_or_init_user_feeded_params_set(self, conf=None):
        """获取或初始化用户已配置参数集合
        用于记录用户实际配置过的所有参数名称
        Args:
            conf: 配置信息，用于从现有配置初始化集合
        """
        if not hasattr(self, _USER_FEEDED_PARAMS):
            if conf is None:
                setattr(self, _USER_FEEDED_PARAMS, set())
            else:
                setattr(self, _USER_FEEDED_PARAMS, set(conf[_USER_FEEDED_PARAMS]))
        return getattr(self, _USER_FEEDED_PARAMS)

    def get_user_feeded(self):
        """获取用户已配置参数集合"""
        return self._get_or_init_user_feeded_params_set()

    def get_feeded_deprecated_params(self):
        """获取已配置的弃用参数集合"""
        return self._get_or_init_feeded_deprecated_params_set()

    @property
    def _deprecated_params_set(self):
        """将弃用参数集合转换为字典格式
        用于快速查找已弃用参数
        """
        return {name: True for name in self.get_feeded_deprecated_params()}

    def __str__(self):
        """将参数对象序列化为JSON字符串"""
        return json.dumps(self.as_dict(), ensure_ascii=False)

    def as_dict(self):
        """将参数对象递归转换为字典结构
        
        功能：
        - 将复杂的参数对象结构转换为纯字典格式
        - 处理嵌套对象、DataFrame等特殊类型
        - 过滤掉内部管理用的属性
        
        返回：
        dict: 转换后的字典结构
        """
        def _recursive_convert_obj_to_dict(obj):
            """递归将对象转换为字典
            
            Args:
                obj: 需要转换的对象
                
            Returns:
                dict: 转换后的字典
            """
            # 初始化返回字典
            ret_dict = {}
            
            # 遍历对象的所有属性
            # 遍历对象的所有属性名称
            # 使用list()将__dict__转换为列表,避免在遍历过程中修改字典导致的问题
            for attr_name in list(obj.__dict__):
                # 跳过内部管理属性（以下划线开头的特殊属性）
                if attr_name in [_FEEDED_DEPRECATED_PARAMS,  # 已配置的弃用参数
                               _DEPRECATED_PARAMS,           # 所有弃用参数
                               _USER_FEEDED_PARAMS,         # 用户配置的参数
                               _IS_RAW_CONF]:               # 是否为原始配置
                    continue
                
                # 获取属性值
                attr = getattr(obj, attr_name)
                
                # 处理不同类型的属性值：
                
                # 1. 如果是DataFrame类型，转换为字典
                if isinstance(attr, pd.DataFrame):
                    ret_dict[attr_name] = attr.to_dict()
                    continue
                
                # 2. 如果是自定义对象（非内置类型）且不为空，递归处理
                if attr and type(attr).__name__ not in dir(builtins):
                    ret_dict[attr_name] = _recursive_convert_obj_to_dict(attr)
                # 3. 如果是内置类型或None，直接存储
                else:
                    ret_dict[attr_name] = attr

            return ret_dict

        # 从self开始递归转换
        return _recursive_convert_obj_to_dict(self)

    def update(self, conf, allow_redundant=False):
        """
        递归更新参数值
        
        Args:
            conf: 配置字典，包含需要更新的参数
            allow_redundant: 是否允许冗余参数（默认False）
        
        Returns:
            更新后的参数对象
        """
        # 检查是否从原始配置更新
        update_from_raw_conf = conf.get(_IS_RAW_CONF, True)
        
        # 处理原始配置的情况
        if update_from_raw_conf:
            # 获取各种参数集合
            deprecated_params_set = self._get_or_init_deprecated_params_set()  # 已弃用参数集合
            feeded_deprecated_params_set = (
                self._get_or_init_feeded_deprecated_params_set()  # 已配置的弃用参数集合
            )
            user_feeded_params_set = self._get_or_init_user_feeded_params_set()  # 用户配置的参数集合
            setattr(self, _IS_RAW_CONF, False)  # 标记非原始配置
        else:
            # 从现有配置更新
            feeded_deprecated_params_set = (
                self._get_or_init_feeded_deprecated_params_set(conf)
            )
            user_feeded_params_set = self._get_or_init_user_feeded_params_set(conf)

        def _recursive_update_param(param, config, depth, prefix):
            """
            递归更新参数
            
            Args:
                param: 要更新的参数对象
                config: 配置字典
                depth: 当前递归深度
                prefix: 参数前缀，用于构建完整参数路径
            """
            # 检查递归深度
            if depth > settings.PARAM_MAXDEPTH:
                raise ValueError("参数嵌套太深，无法解析")

            # 获取当前参数对象的所有属性
            inst_variables = param.__dict__
            redundant_attrs = []  # 记录冗余属性

            # 遍历配置项
            for config_key, config_value in config.items():
                # 处理冗余属性（当前参数对象中不存在的属性）
                if config_key not in inst_variables:
                    if not update_from_raw_conf and config_key.startswith("_"):
                        # 非原始配置且以下划线开头的属性
                        setattr(param, config_key, config_value)
                    else:
                        # 设置新属性
                        setattr(param, config_key, config_value)
                        # redundant_attrs.append(config_key)  # 记录冗余属性（已注释）
                    continue

                # 构建完整的配置键名
                full_config_key = f"{prefix}{config_key}"

                # 处理原始配置的情况
                if update_from_raw_conf:
                    # 记录用户配置的参数
                    user_feeded_params_set.add(full_config_key)

                    # 如果是已弃用参数，添加到已配置的弃用参数集合
                    if full_config_key in deprecated_params_set:
                        feeded_deprecated_params_set.add(full_config_key)

                # 获取当前属性值
                attr = getattr(param, config_key)
                
                # 处理内置类型或None
                if type(attr).__name__ in dir(builtins) or attr is None:
                    setattr(param, config_key, config_value)
                else:
                    # 递归处理自定义对象属性
                    sub_params = _recursive_update_param(
                        attr, config_value, depth + 1, prefix=f"{prefix}{config_key}."
                    )
                    setattr(param, config_key, sub_params)

            # 检查是否允许冗余属性
            if not allow_redundant and redundant_attrs:
                raise ValueError(
                    f"组件 `{getattr(self, '_name', type(self))}` 包含冗余参数: `{[redundant_attrs]}`"
                )

            return param

        # 从当前对象开始递归更新
        return _recursive_update_param(param=self, config=conf, depth=0, prefix="")

    def extract_not_builtin(self):
        def _get_not_builtin_types(obj):
            ret_dict = {}
            for variable in obj.__dict__:
                attr = getattr(obj, variable)
                if attr and type(attr).__name__ not in dir(builtins):
                    ret_dict[variable] = _get_not_builtin_types(attr)

            return ret_dict

        return _get_not_builtin_types(self)

    def validate(self):
        """验证参数值是否符合规则"""
        # 1. 获取所有Python内置类型的名称列表
        self.builtin_types = dir(builtins)
        
        # 2. 定义验证函数字典，将验证操作符映射到对应的验证方法
        self.func = {
            "ge": self._greater_equal_than,    # 大于等于
            "le": self._less_equal_than,       # 小于等于
            "in": self._in,                    # 包含于
            "not_in": self._not_in,           # 不包含于
            "range": self._range,             # 范围检查
        }
        
        # 3. 构建验证规则文件的路径
        # 3.1 获取当前文件的绝对路径
        home_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        # 3.2 构建验证规则文件所在目录的路径
        param_validation_path_prefix = home_dir + "/param_validation/"

        # 4. 获取当前参数类的类名
        param_name = type(self).__name__
        # 4.1 构建完整的验证规则文件路径
        # 例如：如果类名为 MyParam，则文件路径为 .../param_validation/MyParam.json
        param_validation_path = "/".join(
            [param_validation_path_prefix, param_name + ".json"]
        )

        # 5. 初始化验证规则变量
        validation_json = None

        # 6. 尝试读取并解析验证规则文件
        try:
            # 6.1 打开验证规则文件
            with open(param_validation_path, "r") as fin:
                # 6.2 将JSON文件内容解析为Python对象
                validation_json = json.loads(fin.read())
        except BaseException:
            # 6.3 如果文件不存在或解析失败，直接返回
            # 这意味着该参数类没有特定的验证规则
            return

        # 7. 使用加载的验证规则执行参数验证
        # 调用_validate_param方法进行实际的验证工作
        self._validate_param(self, validation_json)

    def _validate_param(self, param_obj, validation_json):
        """递归验证参数对象的所有属性是否符合验证规则
        
        Args:
            param_obj: 需要验证的参数对象
            validation_json: 包含验证规则的JSON对象
        """
        # 1. 获取参数对象的类名，用作验证规则中的节点名
        default_section = type(param_obj).__name__
        
        # 2. 获取参数对象的所有属性列表
        var_list = param_obj.__dict__

        # 3. 遍历所有属性进行验证
        for variable in var_list:
            # 3.1 获取当前属性的值
            attr = getattr(param_obj, variable)

            # 3.2 处理内置类型或None值的情况
            if type(attr).__name__ in self.builtin_types or attr is None:
                # 如果该属性没有对应的验证规则，跳过验证
                if variable not in validation_json:
                    continue

                # 获取该属性的验证规则字典
                # 例如：{"ge": 0, "le": 100} 表示值应该在0到100之间
                validation_dict = validation_json[default_section][variable]
                
                # 重新获取属性值（确保使用最新值）
                value = getattr(param_obj, variable)
                
                # 标记值是否合法的标志
                value_legal = False

                # 遍历所有验证规则
                for op_type in validation_dict:
                    # 使用对应的验证函数检查值是否合法
                    # self.func[op_type]可能是：_greater_equal_than, _less_equal_than等
                    # validation_dict[op_type]是验证的目标值
                    if self.func[op_type](value, validation_dict[op_type]):
                        value_legal = True
                        break

                # 如果所有规则验证后值仍然不合法，抛出异常
                if not value_legal:
                    raise ValueError(
                        "Plase check runtime conf, {} = {} does not match user-parameter restriction".format(
                            variable, value
                        )
                    )

            # 3.3 处理自定义类型的情况（递归验证）
            elif variable in validation_json:
                # 递归调用验证方法，处理嵌套的参数对象
                self._validate_param(attr, validation_json)

    # 各种参数检查的静态方法
    @staticmethod
    def check_string(param, descr):
        """检查参数是否为字符串类型"""
        if type(param).__name__ not in ["str"]:
            raise ValueError(
                descr + " {} not supported, should be string type".format(param)
            )

    @staticmethod
    def check_empty(param, descr):
        """检查参数是否为空"""
        if not param:
            raise ValueError(
                descr + " does not support empty value."
            )

    @staticmethod
    def check_positive_integer(param, descr):
        if type(param).__name__ not in ["int", "long"] or param <= 0:
            raise ValueError(
                descr + " {} not supported, should be positive integer".format(param)
            )

    @staticmethod
    def check_positive_number(param, descr):
        if type(param).__name__ not in ["float", "int", "long"] or param <= 0:
            raise ValueError(
                descr + " {} not supported, should be positive numeric".format(param)
            )

    @staticmethod
    def check_nonnegative_number(param, descr):
        if type(param).__name__ not in ["float", "int", "long"] or param < 0:
            raise ValueError(
                descr
                + " {} not supported, should be non-negative numeric".format(param)
            )

    @staticmethod
    def check_decimal_float(param, descr):
        if type(param).__name__ not in ["float", "int"] or param < 0 or param > 1:
            raise ValueError(
                descr
                + " {} not supported, should be a float number in range [0, 1]".format(
                    param
                )
            )

    @staticmethod
    def check_boolean(param, descr):
        if type(param).__name__ != "bool":
            raise ValueError(
                descr + " {} not supported, should be bool type".format(param)
            )

    @staticmethod
    def check_open_unit_interval(param, descr):
        if type(param).__name__ not in ["float"] or param <= 0 or param >= 1:
            raise ValueError(
                descr + " should be a numeric number between 0 and 1 exclusively"
            )

    @staticmethod
    def check_valid_value(param, descr, valid_values):
        if param not in valid_values:
            raise ValueError(
                descr
                + " {} is not supported, it should be in {}".format(param, valid_values)
            )

    @staticmethod
    def check_defined_type(param, descr, types):
        if type(param).__name__ not in types:
            raise ValueError(
                descr + " {} not supported, should be one of {}".format(param, types)
            )

    @staticmethod
    def check_and_change_lower(param, valid_list, descr=""):
        if type(param).__name__ != "str":
            raise ValueError(
                descr
                + " {} not supported, should be one of {}".format(param, valid_list)
            )

        lower_param = param.lower()
        if lower_param in valid_list:
            return lower_param
        else:
            raise ValueError(
                descr
                + " {} not supported, should be one of {}".format(param, valid_list)
            )

    @staticmethod
    def _greater_equal_than(value, limit):
        return value >= limit - settings.FLOAT_ZERO

    @staticmethod
    def _less_equal_than(value, limit):
        return value <= limit + settings.FLOAT_ZERO

    @staticmethod
    def _range(value, ranges):
        in_range = False
        for left_limit, right_limit in ranges:
            if (
                    left_limit - settings.FLOAT_ZERO
                    <= value
                    <= right_limit + settings.FLOAT_ZERO
            ):
                in_range = True
                break

        return in_range

    @staticmethod
    def _in(value, right_value_list):
        return value in right_value_list

    @staticmethod
    def _not_in(value, wrong_value_list):
        return value not in wrong_value_list

    def _warn_deprecated_param(self, param_name, descr):
        if self._deprecated_params_set.get(param_name):
            logging.warning(
                f"{descr} {param_name} is deprecated and ignored in this version."
            )

    def _warn_to_deprecate_param(self, param_name, descr, new_param):
        if self._deprecated_params_set.get(param_name):
            logging.warning(
                f"{descr} {param_name} will be deprecated in future release; "
                f"please use {new_param} instead."
            )
            return True
        return False


class ComponentBase(ABC):
    """组件的基础类，定义了组件的基本行为和接口"""
    
    component_name: str  # 组件名称

    def __init__(self, canvas, id, param: ComponentParamBase):
        """
        初始化组件
        
        Args:
            canvas: 画布实例
            id: 组件ID
            param: 组件参数
        """
        self._canvas = canvas
        self._id = id
        self._param = param
        self._param.check()

    def get_dependent_components(self):
        """获取依赖的组件列表"""
        cpnts = set([para["component_id"].split("@")[0] for para in self._param.query \
                     if para.get("component_id") \
                     and para["component_id"].lower().find("answer") < 0 \
                     and para["component_id"].lower().find("begin") < 0])
        return list(cpnts)

    def run(self, history, **kwargs):
        """
        运行组件
        
        Args:
            history: 历史记录
            **kwargs: 其他参数
        """
        logging.debug("{}, history: {}, kwargs: {}".format(self, json.dumps(history, ensure_ascii=False),
                                                              json.dumps(kwargs, ensure_ascii=False)))
        self._param.debug_inputs = []
        try:
            res = self._run(history, **kwargs)
            self.set_output(res)
        except Exception as e:
            self.set_output(pd.DataFrame([{"content": str(e)}]))
            raise e

        return res

    def _run(self, history, **kwargs):
        """实际运行逻辑，需要子类实现"""
        raise NotImplementedError()

    def output(self, allow_partial=True):
        """
        获取组件输出
        
        Args:
            allow_partial: 是否允许部分输出，默认为True
                          当为True时允许返回partial对象(延迟执行的函数)
                          当为False时会执行partial对象获取实际结果
        """
        # 获取组件的输出变量值
        o = getattr(self._param, self._param.output_var_name)
        
        # 情况1: 如果输出不是partial对象(即不是延迟执行的函数)
        if not isinstance(o, partial):
            # 1.1 如果输出不是DataFrame格式
            if not isinstance(o, pd.DataFrame):
                # 如果是列表,转换为DataFrame
                if isinstance(o, list):
                    return self._param.output_var_name, pd.DataFrame(o)
                # 如果是None,返回空DataFrame    
                if o is None:
                    return self._param.output_var_name, pd.DataFrame()
                # 其他情况,将输出转为字符串并包装为DataFrame    
                return self._param.output_var_name, pd.DataFrame([{"content": str(o)}])
            # 1.2 如果本身就是DataFrame,直接返回
            return self._param.output_var_name, o

        # 情况2: 如果允许partial输出或输出不是partial对象
        if allow_partial or not isinstance(o, partial):
            # 2.1 如果不是partial且不是DataFrame,转换为DataFrame
            if not isinstance(o, partial) and not isinstance(o, pd.DataFrame):
                return pd.DataFrame(o if isinstance(o, list) else [o])
            # 2.2 其他情况直接返回
            return self._param.output_var_name, o

        # 情况3: 不允许partial输出且输出是partial对象
        outs = None
        # 执行partial对象获取实际结果
        for oo in o():
            # 如果结果不是DataFrame,转换为DataFrame
            if not isinstance(oo, pd.DataFrame):
                outs = pd.DataFrame(oo if isinstance(oo, list) else [oo])
            else:
                outs = oo
        # 返回最终结果        
        return self._param.output_var_name, outs

    def reset(self):
        """重置组件状态"""
        setattr(self._param, self._param.output_var_name, None)
        self._param.inputs = []

    def set_output(self, v):
        """设置组件输出"""
        setattr(self._param, self._param.output_var_name, v)

    def get_input(self):
        """获取组件输入
        
        这个方法用于获取组件的输入数据，包含三种主要的输入来源：
        1. 调试输入 (debug_inputs)
        2. 查询参数输入 (query)
        3. 上游组件输出 (upstream_outs)
        """
        # 1. 首先检查是否有调试输入
        # 如果存在调试输入，优先处理这些输入
        if self._param.debug_inputs:
            # 使用列表推导式处理调试输入：
            # 1. 遍历所有调试输入
            # 2. 只处理包含"value"的输入项
            # 3. 将每个输入转换为标准的DataFrame格式
            return pd.DataFrame([{"content": v["value"]} for v in self._param.debug_inputs if v.get("value")])

        # 2. 获取组件路径
        # 初始化一个空列表用于存储反转的组件路径
        reversed_cpnts = []
        # 如果路径长度大于1，添加倒数第二个路径
        if len(self._canvas.path) > 1:
            reversed_cpnts.extend(self._canvas.path[-2])  # 添加倒数第二个路径
        # 添加最后一个路径
        reversed_cpnts.extend(self._canvas.path[-1])      # 添加最后一个路径

        # 3. 处理查询参数输入
        if self._param.query:
            # 清空现有的输入列表，准备重新填充
            self._param.inputs = []  # 清空现有输入
            # 初始化输出列表，用于存储所有处理结果
            outs = []  # 存储所有输出结果
            
            # 遍历每个查询参数
            for q in self._param.query:
                # 3.1 处理来自其他组件的查询
                if q.get("component_id"):  # 如果查询来自其他组件
                    # 3.1.1 处理begin类型组件
                    if q["component_id"].split("@")[0].lower().find("begin") >= 0:
                        # 解析组件ID和参数键
                        cpn_id, key = q["component_id"].split("@")
                        # 在begin组件中查找对应的参数
                        for p in self._canvas.get_component(cpn_id)["obj"]._param.query:
                            # 如果找到匹配的参数键
                            if p["key"] == key:
                                # 将参数值添加到输出列表
                                outs.append(pd.DataFrame([{"content": p.get("value", "")}]))
                                # 将参数信息添加到输入列表
                                self._param.inputs.append({
                                    "component_id": q["component_id"],
                                    "content": p.get("value", "")
                                })
                                break
                        else:
                            # 如果没有找到匹配的参数，抛出异常
                            assert False, f"Can't find parameter '{key}' for {cpn_id}"
                        continue

                    # 3.1.2 处理answer类型组件
                    if q["component_id"].lower().find("answer") == 0:
                        # 初始化消息文本列表
                        txt = []
                        # 处理历史消息记录
                        for r, c in self._canvas.history[::-1][:self._param.message_history_window_size][::-1]:
                            # 格式化每条消息
                            txt.append(f"{r.upper()}: {c}")
                        # 合并所有消息
                        txt = "\n".join(txt)
                        # 添加到输入列表
                        self._param.inputs.append({
                            "content": txt,
                            "component_id": q["component_id"]
                        })
                        # 添加到输出列表
                        outs.append(pd.DataFrame([{"content": txt}]))
                        continue

                    # 3.1.3 处理其他类型组件
                    # 获取组件输出并添加到输出列表
                    outs.append(self._canvas.get_component(q["component_id"])["obj"].output(allow_partial=False)[1])
                    # 将组件输出添加到输入列表
                    self._param.inputs.append({
                        "component_id": q["component_id"],
                        "content": "\n".join([str(d["content"]) for d in outs[-1].to_dict('records')])
                    })
                
                # 3.2 处理直接值输入
                elif q.get("value"):
                    # 将直接值添加到输入列表
                    self._param.inputs.append({"component_id": None, "content": q["value"]})
                    # 将直接值添加到输出列表
                    outs.append(pd.DataFrame([{"content": q["value"]}]))
                
            # 3.3 合并所有输出结果
            if outs:
                # 合并所有DataFrame
                df = pd.concat(outs, ignore_index=True)
                # 如果存在content列，去除重复内容
                if "content" in df:
                    df = df.drop_duplicates(subset=['content']).reset_index(drop=True)
                return df

        # 4. 处理上游组件输出
        # 初始化上游输出列表
        upstream_outs = []
        
        # 遍历反转的组件路径
        for u in reversed_cpnts[::-1]:
            # 4.1 跳过特殊组件
            if self.get_component_name(u) in ["switch", "concentrator"]:
                continue
            
            # 4.2 处理generate和retrieval组件的特殊情况
            if self.component_name.lower() == "generate" and self.get_component_name(u) == "retrieval":
                o = self._canvas.get_component(u)["obj"].output(allow_partial=False)[1]
                if o is not None:
                    o["component_id"] = u
                    upstream_outs.append(o)
                    continue
                
            # 4.3 跳过relevant和categorize组件
            if self.component_name.lower().find("switch") < 0 \
                    and self.get_component_name(u) in ["relevant", "categorize"]:
                continue
            
            # 4.4 处理answer组件
            if u.lower().find("answer") >= 0:
                for r, c in self._canvas.history[::-1]:
                    if r == "user":
                        upstream_outs.append(pd.DataFrame([{"content": c, "component_id": u}]))
                        break
                break
            
            # 4.5 跳过answer组件的relevant输入
            if self.component_name.lower().find("answer") >= 0 and self.get_component_name(u) in ["relevant"]:
                continue
            
            # 4.6 获取普通组件输出
            o = self._canvas.get_component(u)["obj"].output(allow_partial=False)[1]
            if o is not None:
                o["component_id"] = u
                upstream_outs.append(o)
            break

        # 5. 确保有上游输出
        assert upstream_outs, "Can't inference the where the component input is. Please identify whose output is this component's input."

        # 6. 合并所有上游输出
        df = pd.concat(upstream_outs, ignore_index=True)
        if "content" in df:
            df = df.drop_duplicates(subset=['content']).reset_index(drop=True)

        # 7. 更新输入列表
        self._param.inputs = []
        for _, r in df.iterrows():
            self._param.inputs.append({"component_id": r["component_id"], "content": r["content"]})

        # 8. 返回最终结果
        return df

    def get_input_elements(self):
        """获取输入元素列表"""
        assert self._param.query, "Please identify input parameters firstly."
        eles = []
        for q in self._param.query:
            if q.get("component_id"):
                cpn_id = q["component_id"]
                if cpn_id.split("@")[0].lower().find("begin") >= 0:
                    cpn_id, key = cpn_id.split("@")
                    eles.extend(self._canvas.get_component(cpn_id)["obj"]._param.query)
                    continue

                eles.append({"name": self._canvas.get_compnent_name(cpn_id), "key": cpn_id})
            else:
                eles.append({"key": q["value"], "name": q["value"], "value": q["value"]})
        return eles

    def get_stream_input(self):
        """获取流式输入"""
        reversed_cpnts = []
        if len(self._canvas.path) > 1:
            reversed_cpnts.extend(self._canvas.path[-2])
        reversed_cpnts.extend(self._canvas.path[-1])

        for u in reversed_cpnts[::-1]:
            if self.get_component_name(u) in ["switch", "answer"]:
                continue
            return self._canvas.get_component(u)["obj"].output()[1]

    @staticmethod
    def be_output(v):
        """创建输出DataFrame"""
        return pd.DataFrame([{"content": v}])

    def get_component_name(self, cpn_id):
        """获取组件名称"""
        return self._canvas.get_component(cpn_id)["obj"].component_name.lower()

    def debug(self, **kwargs):
        """调试模式运行"""
        return self._run([], **kwargs)

    def get_parent(self):
        """获取父组件"""
        pid = self._canvas.get_component(self._id)["parent_id"]
        return self._canvas.get_component(pid)["obj"]
