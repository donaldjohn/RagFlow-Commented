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
import os
import requests
from openai.lib.azure import AzureOpenAI
import io
from abc import ABC
from openai import OpenAI
import json
from rag.utils import num_tokens_from_string
import base64
import re


# 语音转文本(ASR)模型模块
# 提供多种语音识别服务的实现，包括OpenAI、通义千问、Azure、Xinference、腾讯云等

# 语音转文本模型的基础抽象类
class Base(ABC):
    """语音转文本模型的基础抽象类
    定义了语音识别模型的基本接口
    """
    def __init__(self, key, model_name):
        """初始化语音识别模型
        Args:
            key: API密钥
            model_name: 模型名称
        """
        pass

    def transcription(self, audio, **kwargs):
        """将音频转换为文本
        Args:
            audio: 音频数据
            **kwargs: 其他参数
        Returns:
            识别的文本和token数量
        """
        transcription = self.client.audio.transcriptions.create(
            model=self.model_name,
            file=audio,
            response_format="text"
        )
        return transcription.text.strip(), num_tokens_from_string(transcription.text.strip())

    def audio2base64(self, audio):
        """将音频文件转换为base64编码
        Args:
            audio: 音频数据(bytes或BytesIO对象)
        Returns:
            base64编码的音频数据
        Raises:
            TypeError: 当输入格式不正确时
        """
        if isinstance(audio, bytes):
            return base64.b64encode(audio).decode("utf-8")
        if isinstance(audio, io.BytesIO):
            return base64.b64encode(audio.getvalue()).decode("utf-8")
        raise TypeError("The input audio file should be in binary format.")


# OpenAI的语音转文本模型实现类
class GPTSeq2txt(Base):
    """OpenAI的语音转文本模型实现
    使用OpenAI API进行语音识别
    """
    def __init__(self, key, model_name="whisper-1", base_url="https://api.openai.com/v1"):
        """初始化OpenAI语音识别模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name


# 通义千问的语音转文本模型实现类
class QWenSeq2txt(Base):
    """通义千问的语音转文本模型实现
    使用通义千问API进行语音识别
    """
    def __init__(self, key, model_name="paraformer-realtime-8k-v1", **kwargs):
        """初始化通义千问语音识别模型
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        import dashscope
        dashscope.api_key = key
        self.model_name = model_name

    def transcription(self, audio, format):
        """执行语音识别
        Args:
            audio: 音频数据
            format: 音频格式
        Returns:
            识别的文本和token数量
        """
        from http import HTTPStatus
        from dashscope.audio.asr import Recognition

        # 创建语音识别实例
        recognition = Recognition(model=self.model_name,
                                  format=format,
                                  sample_rate=16000,
                                  callback=None)
        # 调用API进行语音识别
        result = recognition.call(audio)

        ans = ""
        if result.status_code == HTTPStatus.OK:
            # 处理识别结果
            for sentence in result.get_sentence():
                ans += sentence.text.decode('utf-8') + '\n'
            return ans, num_tokens_from_string(ans)

        return "**ERROR**: " + result.message, 0


# Azure的语音转文本模型实现类
class AzureSeq2txt(Base):
    """Azure的语音转文本模型实现
    使用Azure API进行语音识别
    """
    def __init__(self, key, model_name, lang="Chinese", **kwargs):
        """初始化Azure语音识别模型
        Args:
            key: API密钥
            model_name: 模型名称
            lang: 语言设置
            **kwargs: 其他参数
        """
        self.client = AzureOpenAI(api_key=key, azure_endpoint=kwargs["base_url"], api_version="2024-02-01")
        self.model_name = model_name
        self.lang = lang


# Xinference的语音转文本模型实现类
class XinferenceSeq2txt(Base):
    """Xinference的语音转文本模型实现
    使用Xinference API进行语音识别
    """
    def __init__(self, key, model_name="whisper-small", **kwargs):
        """初始化Xinference语音识别模型
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        self.base_url = kwargs.get('base_url', None)
        self.model_name = model_name
        self.key = key

    def transcription(self, audio, language="zh", prompt=None, response_format="json", temperature=0.7):
        """执行语音识别
        Args:
            audio: 音频数据或文件路径
            language: 语言设置
            prompt: 提示文本
            response_format: 响应格式
            temperature: 温度参数
        Returns:
            识别的文本和token数量
        """
        # 处理音频文件
        if isinstance(audio, str):
            audio_file = open(audio, 'rb')
            audio_data = audio_file.read()
            audio_file_name = audio.split("/")[-1]
        else:
            audio_data = audio
            audio_file_name = "audio.wav"

        # 准备请求参数
        payload = {
            "model": self.model_name,
            "language": language,
            "prompt": prompt,
            "response_format": response_format,
            "temperature": temperature
        }

        files = {
            "file": (audio_file_name, audio_data, 'audio/wav')
        }

        try:
            # 调用API进行语音识别
            response = requests.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files=files,
                data=payload
            )
            response.raise_for_status()
            result = response.json()

            if 'text' in result:
                transcription_text = result['text'].strip()
                return transcription_text, num_tokens_from_string(transcription_text)
            else:
                return "**ERROR**: Failed to retrieve transcription.", 0

        except requests.exceptions.RequestException as e:
            return f"**ERROR**: {str(e)}", 0


# 腾讯云的语音转文本模型实现类
class TencentCloudSeq2txt(Base):
    """腾讯云的语音转文本模型实现
    使用腾讯云API进行语音识别
    """
    def __init__(
            self, key, model_name="16k_zh", base_url="https://asr.tencentcloudapi.com"
    ):
        """初始化腾讯云语音识别模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        from tencentcloud.common import credential
        from tencentcloud.asr.v20190614 import asr_client

        # 初始化腾讯云客户端
        key = json.loads(key)
        sid = key.get("tencent_cloud_sid", "")
        sk = key.get("tencent_cloud_sk", "")
        cred = credential.Credential(sid, sk)
        self.client = asr_client.AsrClient(cred, "")
        self.model_name = model_name

    def transcription(self, audio, max_retries=60, retry_interval=5):
        """执行语音识别
        Args:
            audio: 音频数据
            max_retries: 最大重试次数
            retry_interval: 重试间隔(秒)
        Returns:
            识别的文本和token数量
        """
        from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
            TencentCloudSDKException,
        )
        from tencentcloud.asr.v20190614 import models
        import time

        # 将音频转换为base64
        b64 = self.audio2base64(audio)
        try:
            # 创建识别任务
            req = models.CreateRecTaskRequest()
            params = {
                "EngineModelType": self.model_name,
                "ChannelNum": 1,
                "ResTextFormat": 0,
                "SourceType": 1,
                "Data": b64,
            }
            req.from_json_string(json.dumps(params))
            resp = self.client.CreateRecTask(req)

            # 轮询查询任务状态
            req = models.DescribeTaskStatusRequest()
            params = {"TaskId": resp.Data.TaskId}
            req.from_json_string(json.dumps(params))
            retries = 0
            while retries < max_retries:
                resp = self.client.DescribeTaskStatus(req)
                if resp.Data.StatusStr == "success":
                    # 处理识别结果，移除时间戳标记
                    text = re.sub(
                        r"\[\d+:\d+\.\d+,\d+:\d+\.\d+\]\s*", "", resp.Data.Result
                    ).strip()
                    return text, num_tokens_from_string(text)
                elif resp.Data.StatusStr == "failed":
                    return (
                        "**ERROR**: Failed to retrieve speech recognition results.",
                        0,
                    )
                else:
                    time.sleep(retry_interval)
                    retries += 1
            return "**ERROR**: Max retries exceeded. Task may still be processing.", 0

        except TencentCloudSDKException as e:
            return "**ERROR**: " + str(e), 0
        except Exception as e:
            return "**ERROR**: " + str(e), 0


# GPUStack的语音转文本模型实现类
class GPUStackSeq2txt(Base):
    """GPUStack的语音转文本模型实现
    使用GPUStack API进行语音识别
    """
    def __init__(self, key, model_name, base_url):
        """初始化GPUStack语音识别模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        Raises:
            ValueError: 当base_url为空时
        """
        if not base_url:
            raise ValueError("url cannot be None")
        if base_url.split("/")[-1] != "v1-openai":
            base_url = os.path.join(base_url, "v1-openai")
        self.base_url = base_url
        self.model_name = model_name
        self.key = key
