<!-- <div align="center">
<h1>
  Zhigong
</h1>
</div> -->

<div align="center"><img src="misc/logo.png" width="550"/></div>


<p align="center">
  👨‍💻 <a href="https://github.com/ciictec/Zhigong1.6B" target="_blank">Github</a> • 🤗 <a href="https://huggingface.co/ciictec" target="_blank">Hugging Face</a>
</p>


# 项目介绍
Zhigong-1.6B是工业领域轻量化开源大模型，采用3.2T的高质量语料训练。
- 该底座产品面向边缘计算和智能终端，1.6B参数实现了模型参数的轻量级化。该底座产品，提供高度灵活的预训练框架，可以将能力扩展至工业设备、智能设备和工业产品，为工业应用场景提供更高效的计算性能。
- 该底座产品实现了对汉语、英语、法语、俄语、西班牙语、柬埔寨语、捷克语、匈牙利语、印度尼西亚语、哈萨克语、老挝语、蒙古语、波兰语、塞尔维亚语、土耳其语、越南语、日语和韩语等18种语言文字语言增强技术。

# 更新信息
* 2023.12.28 智工开源1.6B工业大模型已开放。地址：[huggingface](https://huggingface.co/ciictec)

# 目录

- [下载地址](#下载地址)
- [模型介绍](#模型介绍)
- [模型评估](#模型评估)
- [快速开始](#快速开始)
- [模型样例展示](#模型样例展示)
<!-- - [声明和协议](#声明和协议) -->
<!-- - [引用和联系我们](#引用和联系我们) -->


# 下载地址
## 模型下载
| 模型 | 下载地址 |
| --------- | ------ | 
| Zhigong-1.6B-Base     |  🤗[Zhigong-1.6B-Base](https://huggingface.co/ciictec/Zhigong-1.6B-Base)|
| Zhigong-1.6B-Chat     |  🤗[Zhigong-1.6B-Chat](https://huggingface.co/ciictec/Zhigong-1.6B-Chat)|



# 模型介绍

## 训练数据
我们针对工业场景重新构建了数据筛选流程，Zhigong-1.6B-Base模型是在清洗后的3.2T高质量中、英、代码数据上进行训练，大幅度提升了图书、论文、领域数据比例。

|  | 类目     | 百分比    |
| ---------------------------- | ------ | ------ |
| 英文                           | 网页数据   | 36.20% |
|                              | 书籍数据   | 8.70%  |
|                              | 工业领域数据 | 2.60%  |
|                              | 学术论文   | 2.50%  |
|                              | 百科     | 0.50%  |
|                              | 其他     | 1.90%  |
| 中文                           | 网页数据   | 25.20% |
|                              | 书籍数据   | 6.50%  |
|                              | 工业领域数据 | 2.50%  |
|                              | 百科全书   | 0.80%  |
|                              | 其他     | 1.60%  |
| 其他语言                         | 百科     | 2.40%  |
|                              | 图书     | 2.60%  |
| 代码                           | Github | 6.00%  |

## 模型结构
整体模型基于标准的 Transformer 结构，我们采用了和 LLaMA-2相同的模型设计。层数为24层，Hidden Dim 为2048，训练序列长度为2048。
| 模型结构              | **Zhigong-1.6B** |
| ----------------- | ------------ |
| 词表大小              | 103368       |
| Hidden Dim        | 2048         |
| Intermediate Size | 5632         |
| Head Dim          | 128          |
| Attention头数       | 16           |
| 层数                | 24           |
| 训练序列长度            | 2048         |
| 位置编码              | RoPE      |

## 分词器 
我们使用Byte-Pair Encoding（BPE）对数据进行分词。Zhigong-1.6B聚焦“一带一路”国际合作高峰论坛，支持汉语、英语、法语、俄语、西班牙语、柬埔寨语、捷克语、匈牙利语、印度尼西亚语、哈萨克语、老挝语、蒙古语、波兰语、塞尔维亚语、土耳其语、越南语、日语和韩语。并且针对除英语外的“一带一路”国家语言额外引入1万个词元。


# 模型评估
## MMLU评估
| Model                 | MMLU  |
| ------------------------------------------------- | ----- |
| Gpt-neo-1.3B                                      | 24.51 |
| Opt-1.3b                                          | 24.88 |
| TinyLlama-1.1B-intermediate-step-1195k-token-2.5T | 25.92 |
| Pythia-1b                                         | 26.21 |
| Bloom-3b                                          | 26.3  |
| Bloom-1b1                                         | 26.6  |
| Bloomz-1b1                                        | 27.05 |
| Bloom-1b7                                         | 27.59 |
| **Zhigong-1.6B**                                       | **28.01** |

# 快速开始
我们将模型参数、配置文件、tokenizer等在Hugging Face上进行了开源。

## 依赖安装
- Python 3.8及以上版本
- Pytorch 1.13及以上版本
- CUDA建议使用11.4以上版本

运行下面的脚本进行Python依赖安装。
```shell
pip install -r requirements.txt 
```

## Base模型推理
```
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

model_path = "ciictec/Zhigong-1.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, add_bos_token=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()

inputs = tokenizer('黄河发源于青藏高原', return_tensors='pt').to(model.device)
response = model.generate(inputs.input_ids, max_length=128)
print(tokenizer.decode(response.cpu()[0], skip_special_tokens=True))
"""
黄河发源于青藏高原的世界屋脊，是中国的“第二条大河”。它流经9个省、自治区、直辖市，流域面积约960万平方千米，占全国面积的71%，为中国的“水塔”。
它也是世界上水量最大、流域面积最广的河流，也是世界上水资源最丰富的河流之一。
它也是中国的“生命之河”，被誉为“中华水塔”。
"""
```

## Chat模型推理
```
import os
import re
from typing import Tuple, List, Dict
import torch
from loguru import logger
from transformers import AutoTokenizer, LlamaForCausalLM

model_name = "Assistant"
human_prompt = "<|Human|>"
assistant_prompt = f"<|{model_name}|>"

class ModelChat(LlamaForCausalLM):
    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": [human_prompt, assistant_prompt]})
        self.history = []

    def process_response(self, response):
        response = response.strip()
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        response = response.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.bos_token, "").strip()
        return response

    @torch.no_grad()
    def chat(self, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, top_k=50, temperature=0.8, repetition_penalty=1.0, **kwargs):
        if history is None:
            history = []
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "top_k": top_k, "temperature": temperature, "repetition_penalty": repetition_penalty,
                      "bos_token_id": self.tokenizer.bos_token_id, "eos_token_id": self.tokenizer.eos_token_id, **kwargs}
        history_prompt = ""
        for turn in history:
            history_prompt += turn[0] + turn[1]
        wrap_query = human_prompt + str(query)
        prompt = history_prompt + wrap_query + assistant_prompt 
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(outputs)
        response = self.process_response(response)
        history.append([wrap_query, assistant_prompt + response + self.tokenizer.eos_token])
        
        return response, history

if __name__ == '__main__':
    model_path = "ciictec/Zhigong-1.6B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, add_bos_token=False)
    model = ModelChat.from_pretrained(model_path, tokenizer=tokenizer, device_map="auto").half()

    q = "<<桃花源记>>里陶渊明表达了什么样的思想"
    response, history = model.chat(query=q, max_length=2048,history=None)
    print(response)

    """生成结果：
    逃避现实。陶渊明在《桃花源记》中，以“桃花源”为象征，描绘了一个与现实社会截然不同的理想社会，在这个社会中，人们过着快乐、安宁的生活，与自然和谐相处，人们不关心政治，不追求名利，过着与世无争的生活。陶渊明在这里所表达的思想，是逃避现实、追求宁静的思想。
    """
```


# 模型样例展示
我们构建了多种应用场景，使用Zhigong-1.6B-Chat模型进行了测试。

## 工业问答
```
用户：
什么是PLC？
Zhigong-1.6B-Chat：
PLC（Programmable Logic Controller）是一种工业控制系统，它将计算机与工业控制系统相结合，实现对工业设备的控制和操作。它通过编程实现对设备的控制，实现对设备的智能化控制。

用户：
什么是工业自动化的传统组件？
Zhigong-1.6B-Chat：
工业自动化传统组件是指工业自动化系统的基本组成部分，包括传感器、执行器、控制系统、计算机等。这些组件通常由工业制造商或第三方供应商提供，以确保工业自动化系统的稳定性和可靠性。

用户：
什么是工业物联网（IIoT）？
Zhigong-1.6B-Chat：
工业物联网（IIoT）是一种将工业设备、传感器、软件和数据等技术连接起来的网络，以实现工业设备的远程监控、数据采集和智能控制。IIoT可以帮助企业提高生产效率、降低成本、提高产品质量和服务水平，并实现设备的智能维护和故障预测。

用户：
工业中的能源管理是什么？
Zhigong-1.6B-Chat：
工业中的能源管理是指在工业生产过程中，对能源的使用、管理、分配和节约等方面进行的规划、设计、实施和监控等一系列活动。它涉及到生产过程中的各个环节，包括生产设备、生产工艺、生产管理、能源管理、能源利用效率等。
```


## 翻译
```
用户：
请将以下句子翻译成中文：ハワイのweather是very的好，우리 함께 go shopping和ピクニく吧！

Zhigong-1.6B-Chat：
夏威夷的天气很好，我们一起去购物吧！

用户：
"Вечер тихой песнею над рекой плывёт."是什么意思？

Zhigong-1.6B-Chat：
“晚风轻拂，在河流的另一边，我听到那轻快的歌声。”
```

## 代码
```
用户：
写一段C++代码 完成冒泡排序

Zhigong-1.6B-Chat：
#include <iostream>
#include <vector>

void bubbleSort(std::vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++) {
        for (int j = 0; j < arr.size() - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    std::vector<int> arr = {10, 7, 8, 9, 1, 5, 6, 4, 3, 2};
    bubbleSort(arr);
    std::cout << "Sorted array: " << std::endl;
    std::cout << arr << std::endl;
    return 0;
}
```

## 数学
```
用户：
如果3x+1=10，那么x等于多少？

Zhigong-1.6B-Chat：
如果3x+1=10，那么我们可以通过将等式两边都减去1来解出x的值。
3x + 1 - 1 = 10 - 1
3x = 9
现在我们可以解出x的值：
x = 9 / 3
x = 3
所以，x等于3

用户：
序列1,4,9,16的后一个数是什么？

Zhigong-1.6B-Chat：
序列1,4,9,16的后一个数是25。
```

## 诗词文学
```
用户：
“但愿人长久，千里共婵娟”，其中婵娟指的是什么？

Zhigong-1.6B-Chat：
婵娟指的是月亮。

用户：
“不是东风压了西风，就是西风压了东风”一语出自《红楼梦》中何人之口？

Zhigong-1.6B-Chat：
林黛玉。不是东风压了西风，就是西风压了东风。这句出自《红楼梦》中林黛玉的口中。
```

