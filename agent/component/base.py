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
        self.builtin_types = dir(builtins)
        self.func = {
            "ge": self._greater_equal_than,
            "le": self._less_equal_than,
            "in": self._in,
            "not_in": self._not_in,
            "range": self._range,
        }
        home_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        param_validation_path_prefix = home_dir + "/param_validation/"

        param_name = type(self).__name__
        param_validation_path = "/".join(
            [param_validation_path_prefix, param_name + ".json"]
        )

        validation_json = None

        try:
            with open(param_validation_path, "r") as fin:
                validation_json = json.loads(fin.read())
        except BaseException:
            return

        self._validate_param(self, validation_json)

    def _validate_param(self, param_obj, validation_json):
        default_section = type(param_obj).__name__
        var_list = param_obj.__dict__

        for variable in var_list:
            attr = getattr(param_obj, variable)

            if type(attr).__name__ in self.builtin_types or attr is None:
                if variable not in validation_json:
                    continue

                validation_dict = validation_json[default_section][variable]
                value = getattr(param_obj, variable)
                value_legal = False

                for op_type in validation_dict:
                    if self.func[op_type](value, validation_dict[op_type]):
                        value_legal = True
                        break

                if not value_legal:
                    raise ValueError(
                        "Plase check runtime conf, {} = {} does not match user-parameter restriction".format(
                            variable, value
                        )
                    )

            elif variable in validation_json:
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
            allow_partial: 是否允许部分输出
        """
        o = getattr(self._param, self._param.output_var_name)
        if not isinstance(o, partial):
            if not isinstance(o, pd.DataFrame):
                if isinstance(o, list):
                    return self._param.output_var_name, pd.DataFrame(o)
                if o is None:
                    return self._param.output_var_name, pd.DataFrame()
                return self._param.output_var_name, pd.DataFrame([{"content": str(o)}])
            return self._param.output_var_name, o

        if allow_partial or not isinstance(o, partial):
            if not isinstance(o, partial) and not isinstance(o, pd.DataFrame):
                return pd.DataFrame(o if isinstance(o, list) else [o])
            return self._param.output_var_name, o

        outs = None
        for oo in o():
            if not isinstance(oo, pd.DataFrame):
                outs = pd.DataFrame(oo if isinstance(oo, list) else [oo])
            else:
                outs = oo
        return self._param.output_var_name, outs

    def reset(self):
        """重置组件状态"""
        setattr(self._param, self._param.output_var_name, None)
        self._param.inputs = []

    def set_output(self, v):
        """设置组件输出"""
        setattr(self._param, self._param.output_var_name, v)

    def get_input(self):
        """获取组件输入"""
        if self._param.debug_inputs:
            return pd.DataFrame([{"content": v["value"]} for v in self._param.debug_inputs if v.get("value")])

        reversed_cpnts = []
        if len(self._canvas.path) > 1:
            reversed_cpnts.extend(self._canvas.path[-2])
        reversed_cpnts.extend(self._canvas.path[-1])

        if self._param.query:
            self._param.inputs = []
            outs = []
            for q in self._param.query:
                if q.get("component_id"):
                    if q["component_id"].split("@")[0].lower().find("begin") >= 0:
                        cpn_id, key = q["component_id"].split("@")
                        for p in self._canvas.get_component(cpn_id)["obj"]._param.query:
                            if p["key"] == key:
                                outs.append(pd.DataFrame([{"content": p.get("value", "")}]))
                                self._param.inputs.append({"component_id": q["component_id"],
                                                           "content": p.get("value", "")})
                                break
                        else:
                            assert False, f"Can't find parameter '{key}' for {cpn_id}"
                        continue

                    if q["component_id"].lower().find("answer") == 0:
                        txt = []
                        for r, c in self._canvas.history[::-1][:self._param.message_history_window_size][::-1]:
                            txt.append(f"{r.upper()}: {c}")
                        txt = "\n".join(txt)
                        self._param.inputs.append({"content": txt, "component_id": q["component_id"]})
                        outs.append(pd.DataFrame([{"content": txt}]))
                        continue

                    outs.append(self._canvas.get_component(q["component_id"])["obj"].output(allow_partial=False)[1])
                    self._param.inputs.append({"component_id": q["component_id"],
                                               "content": "\n".join(
                                                   [str(d["content"]) for d in outs[-1].to_dict('records')])})
                elif q.get("value"):
                    self._param.inputs.append({"component_id": None, "content": q["value"]})
                    outs.append(pd.DataFrame([{"content": q["value"]}]))
            if outs:
                df = pd.concat(outs, ignore_index=True)
                if "content" in df:
                    df = df.drop_duplicates(subset=['content']).reset_index(drop=True)
                return df

        upstream_outs = []

        for u in reversed_cpnts[::-1]:
            if self.get_component_name(u) in ["switch", "concentrator"]:
                continue
            if self.component_name.lower() == "generate" and self.get_component_name(u) == "retrieval":
                o = self._canvas.get_component(u)["obj"].output(allow_partial=False)[1]
                if o is not None:
                    o["component_id"] = u
                    upstream_outs.append(o)
                    continue
            #if self.component_name.lower()!="answer" and u not in self._canvas.get_component(self._id)["upstream"]: continue
            if self.component_name.lower().find("switch") < 0 \
                    and self.get_component_name(u) in ["relevant", "categorize"]:
                continue
            if u.lower().find("answer") >= 0:
                for r, c in self._canvas.history[::-1]:
                    if r == "user":
                        upstream_outs.append(pd.DataFrame([{"content": c, "component_id": u}]))
                        break
                break
            if self.component_name.lower().find("answer") >= 0 and self.get_component_name(u) in ["relevant"]:
                continue
            o = self._canvas.get_component(u)["obj"].output(allow_partial=False)[1]
            if o is not None:
                o["component_id"] = u
                upstream_outs.append(o)
            break

        assert upstream_outs, "Can't inference the where the component input is. Please identify whose output is this component's input."

        df = pd.concat(upstream_outs, ignore_index=True)
        if "content" in df:
            df = df.drop_duplicates(subset=['content']).reset_index(drop=True)

        self._param.inputs = []
        for _, r in df.iterrows():
            self._param.inputs.append({"component_id": r["component_id"], "content": r["content"]})

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
