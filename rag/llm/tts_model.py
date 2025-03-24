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

import _thread as thread
import base64
import hashlib
import hmac
import json
import queue
import re
import ssl
import time
from abc import ABC
from datetime import datetime
from time import mktime
from typing import Annotated, Literal
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import httpx
import ormsgpack
import requests
import websocket
from pydantic import BaseModel, conint

from rag.utils import num_tokens_from_string


# 用于存储参考音频的数据模型
class ServeReferenceAudio(BaseModel):
    """参考音频数据模型
    Attributes:
        audio: 音频数据
        text: 对应的文本
    """
    audio: bytes  # 音频数据
    text: str     # 对应的文本


# TTS请求的数据模型
class ServeTTSRequest(BaseModel):
    """TTS请求数据模型
    Attributes:
        text: 要转换的文本
        chunk_length: 分块长度(100-300)
        format: 输出音频格式(wav/pcm/mp3)
        mp3_bitrate: MP3比特率(64/128/192)
        references: 参考音频列表
        reference_id: 参考音频ID
        normalize: 是否标准化文本
        latency: 延迟模式(normal/balanced)
    """
    text: str     # 要转换的文本
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200  # 分块长度
    format: Literal["wav", "pcm", "mp3"] = "mp3"  # 输出音频格式
    mp3_bitrate: Literal[64, 128, 192] = 128     # MP3比特率
    references: list[ServeReferenceAudio] = []    # 参考音频列表
    reference_id: str | None = None              # 参考音频ID
    normalize: bool = True                       # 是否标准化文本
    latency: Literal["normal", "balanced"] = "normal"  # 延迟模式


# TTS模型的基础抽象类
class Base(ABC):
    """TTS模型的基础抽象类
    定义了TTS模型的基本接口
    """
    def __init__(self, key, model_name, base_url):
        """初始化TTS模型
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        pass

    def tts(self, audio):
        """文本转语音
        Args:
            audio: 输入文本
        Returns:
            音频数据生成器
        """
        pass

    def normalize_text(self, text):
        """标准化文本，移除特殊标记
        Args:
            text: 输入文本
        Returns:
            标准化后的文本
        """
        return re.sub(r'(\*\*|##\d+\$\$|#)', '', text)


# FishAudio的TTS模型实现类
class FishAudioTTS(Base):
    """FishAudio的TTS模型实现
    使用FishAudio API进行文本转语音
    """
    def __init__(self, key, model_name, base_url="https://api.fish.audio/v1/tts"):
        """初始化FishAudio TTS
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if not base_url:
            base_url = "https://api.fish.audio/v1/tts"
        key = json.loads(key)
        self.headers = {
            "api-key": key.get("fish_audio_ak"),
            "content-type": "application/msgpack",
        }
        self.ref_id = key.get("fish_audio_refid")
        self.base_url = base_url

    def tts(self, text):
        """执行文本转语音
        Args:
            text: 输入文本
        Returns:
            音频数据生成器
        """
        from http import HTTPStatus

        # 标准化文本
        text = self.normalize_text(text)
        request = ServeTTSRequest(text=text, reference_id=self.ref_id)

        # 使用httpx客户端发送请求
        with httpx.Client() as client:
            try:
                with client.stream(
                        method="POST",
                        url=self.base_url,
                        content=ormsgpack.packb(
                            request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC
                        ),
                        headers=self.headers,
                        timeout=None,
                ) as response:
                    if response.status_code == HTTPStatus.OK:
                        # 流式返回音频数据
                        for chunk in response.iter_bytes():
                            yield chunk
                    else:
                        response.raise_for_status()

                yield num_tokens_from_string(text)

            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"**ERROR**: {e}")


# 通义千问的TTS模型实现类
class QwenTTS(Base):
    """通义千问的TTS模型实现
    使用通义千问API进行文本转语音
    """
    def __init__(self, key, model_name, base_url=""):
        """初始化通义千问TTS
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        import dashscope

        self.model_name = model_name
        dashscope.api_key = key

    def tts(self, text):
        """执行文本转语音
        Args:
            text: 输入文本
        Returns:
            音频数据生成器
        """
        from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
        from dashscope.audio.tts import ResultCallback, SpeechSynthesizer, SpeechSynthesisResult
        from collections import deque

        # 定义回调类处理音频数据
        class Callback(ResultCallback):
            """音频数据回调处理类"""
            def __init__(self) -> None:
                self.dque = deque()

            def _run(self):
                """运行音频数据生成器"""
                while True:
                    if not self.dque:
                        time.sleep(0)
                        continue
                    val = self.dque.popleft()
                    if val:
                        yield val
                    else:
                        break

            def on_open(self):
                """连接打开时的回调"""
                pass

            def on_complete(self):
                """合成完成时的回调"""
                self.dque.append(None)

            def on_error(self, response: SpeechSynthesisResponse):
                """发生错误时的回调"""
                raise RuntimeError(str(response))

            def on_close(self):
                """连接关闭时的回调"""
                pass

            def on_event(self, result: SpeechSynthesisResult):
                """收到音频数据时的回调"""
                if result.get_audio_frame() is not None:
                    self.dque.append(result.get_audio_frame())

        # 标准化文本并开始合成
        text = self.normalize_text(text)
        callback = Callback()
        SpeechSynthesizer.call(model=self.model_name,
                               text=text,
                               callback=callback,
                               format="mp3")
        try:
            # 流式返回音频数据
            for data in callback._run():
                yield data
            yield num_tokens_from_string(text)

        except Exception as e:
            raise RuntimeError(f"**ERROR**: {e}")


# OpenAI的TTS模型实现类
class OpenAITTS(Base):
    """OpenAI的TTS模型实现
    使用OpenAI API进行文本转语音
    """
    def __init__(self, key, model_name="tts-1", base_url="https://api.openai.com/v1"):
        """初始化OpenAI TTS
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.api_key = key
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def tts(self, text, voice="alloy"):
        """执行文本转语音
        Args:
            text: 输入文本
            voice: 语音类型
        Returns:
            音频数据生成器
        """
        # 标准化文本
        text = self.normalize_text(text)
        payload = {
            "model": self.model_name,
            "voice": voice,
            "input": text
        }

        # 发送请求并流式返回音频数据
        response = requests.post(f"{self.base_url}/audio/speech", headers=self.headers, json=payload, stream=True)

        if response.status_code != 200:
            raise Exception(f"**Error**: {response.status_code}, {response.text}")
        for chunk in response.iter_content():
            if chunk:
                yield chunk


# 讯飞星火的TTS模型实现类
class SparkTTS:
    """讯飞星火的TTS模型实现
    使用讯飞星火API进行文本转语音
    """
    STATUS_FIRST_FRAME = 0
    STATUS_CONTINUE_FRAME = 1
    STATUS_LAST_FRAME = 2

    def __init__(self, key, model_name, base_url=""):
        """初始化讯飞星火TTS
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        key = json.loads(key)
        self.APPID = key.get("spark_app_id", "xxxxxxx")
        self.APISecret = key.get("spark_api_secret", "xxxxxxx")
        self.APIKey = key.get("spark_api_key", "xxxxxx")
        self.model_name = model_name
        self.CommonArgs = {"app_id": self.APPID}
        self.audio_queue = queue.Queue()

    def create_url(self):
        """生成WebSocket URL
        Returns:
            WebSocket连接URL
        """
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        url = url + '?' + urlencode(v)
        return url

    def tts(self, text):
        """执行文本转语音
        Args:
            text: 输入文本
        Returns:
            音频数据生成器
        """
        # 准备业务参数
        BusinessArgs = {"aue": "lame", "sfl": 1, "auf": "audio/L16;rate=16000", "vcn": self.model_name, "tte": "utf8"}
        Data = {"status": 2, "text": base64.b64encode(text.encode('utf-8')).decode('utf-8')}
        CommonArgs = {"app_id": self.APPID}
        audio_queue = self.audio_queue
        model_name = self.model_name

        # 定义WebSocket回调类
        class Callback:
            """WebSocket回调处理类"""
            def __init__(self):
                self.audio_queue = audio_queue

            def on_message(self, ws, message):
                """收到消息时的回调"""
                message = json.loads(message)
                code = message["code"]
                sid = message["sid"]
                audio = message["data"]["audio"]
                audio = base64.b64decode(audio)
                status = message["data"]["status"]
                if status == 2:
                    ws.close()
                if code != 0:
                    errMsg = message["message"]
                    raise Exception(f"sid:{sid} call error:{errMsg} code:{code}")
                else:
                    self.audio_queue.put(audio)

            def on_error(self, ws, error):
                """发生错误时的回调"""
                raise Exception(error)

            def on_close(self, ws, close_status_code, close_msg):
                """连接关闭时的回调"""
                self.audio_queue.put(None)  # 放入 None 作为结束标志

            def on_open(self, ws):
                """连接打开时的回调"""
                def run(*args):
                    d = {"common": CommonArgs,
                         "business": BusinessArgs,
                         "data": Data}
                    ws.send(json.dumps(d))

                thread.start_new_thread(run, ())

        # 建立WebSocket连接并处理音频数据
        wsUrl = self.create_url()
        websocket.enableTrace(False)
        a = Callback()
        ws = websocket.WebSocketApp(wsUrl, on_open=a.on_open, on_error=a.on_error, on_close=a.on_close,
                                    on_message=a.on_message)
        status_code = 0
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        while True:
            audio_chunk = self.audio_queue.get()
            if audio_chunk is None:
                if status_code == 0:
                    raise Exception(
                        f"Fail to access model({model_name}) using the provided credentials. **ERROR**: Invalid APPID, API Secret, or API Key.")
                else:
                    break
            status_code = 1
            yield audio_chunk


# Xinference的TTS模型实现类
class XinferenceTTS:
    """Xinference的TTS模型实现
    使用Xinference API进行文本转语音
    """
    def __init__(self, key, model_name, **kwargs):
        """初始化Xinference TTS
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        self.base_url = kwargs.get("base_url", None)
        self.model_name = model_name
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

    def tts(self, text, voice="中文女", stream=True):
        """执行文本转语音
        Args:
            text: 输入文本
            voice: 语音类型
            stream: 是否流式返回
        Returns:
            音频数据生成器
        """
        # 准备请求参数
        payload = {
            "model": self.model_name,
            "input": text,
            "voice": voice
        }

        # 发送请求并流式返回音频数据
        response = requests.post(
            f"{self.base_url}/v1/audio/speech",
            headers=self.headers,
            json=payload,
            stream=stream
        )

        if response.status_code != 200:
            raise Exception(f"**Error**: {response.status_code}, {response.text}")

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                yield chunk


# Ollama的TTS模型实现类
class OllamaTTS(Base):
    """Ollama的TTS模型实现
    使用Ollama API进行文本转语音
    """
    def __init__(self, key, model_name="ollama-tts", base_url="https://api.ollama.ai/v1"):
        """初始化Ollama TTS
        Args:
            key: API密钥
            model_name: 模型名称
            base_url: API基础URL
        """
        if not base_url: 
            base_url = "https://api.ollama.ai/v1"
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }

    def tts(self, text, voice="standard-voice"):
        """执行文本转语音
        Args:
            text: 输入文本
            voice: 语音类型
        Returns:
            音频数据生成器
        """
        # 准备请求参数
        payload = {
            "model": self.model_name,
            "voice": voice,
            "input": text
        }

        # 发送请求并流式返回音频数据
        response = requests.post(f"{self.base_url}/audio/tts", headers=self.headers, json=payload, stream=True)

        if response.status_code != 200:
            raise Exception(f"**Error**: {response.status_code}, {response.text}")

        for chunk in response.iter_content():
            if chunk:
                yield chunk


# GPUStack的TTS模型实现类
class GPUStackTTS:
    """GPUStack的TTS模型实现
    使用GPUStack API进行文本转语音
    """
    def __init__(self, key, model_name, **kwargs):
        """初始化GPUStack TTS
        Args:
            key: API密钥
            model_name: 模型名称
            **kwargs: 其他参数
        """
        self.base_url = kwargs.get("base_url", None)
        self.api_key = key
        self.model_name = model_name
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def tts(self, text, voice="Chinese Female", stream=True):
        """执行文本转语音
        Args:
            text: 输入文本
            voice: 语音类型
            stream: 是否流式返回
        Returns:
            音频数据生成器
        """
        # 准备请求参数
        payload = {
            "model": self.model_name,
            "input": text,
            "voice": voice
        }

        # 发送请求并流式返回音频数据
        response = requests.post(
            f"{self.base_url}/v1-openai/audio/speech",
            headers=self.headers,
            json=payload,
            stream=stream
        )

        if response.status_code != 200:
            raise Exception(f"**Error**: {response.status_code}, {response.text}")

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                yield chunk