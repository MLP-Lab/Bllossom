---
language:
- en
- ko
license: apache-2.0
library_name: transformers
base_model:
- meta-llama/Meta-Llama-3-8B
---

---
language:
- en
- ko
license: llama3
library_name: transformers
base_model:
- meta-llama/Meta-Llama-3-8B
---

<a href="https://github.com/MLP-Lab/Bllossom">
  <img src="https://github.com/teddysum/bllossom/blob/main//bllossom_icon.png?raw=true" width="40%" height="50%">
</a>

# Bllossom | [Demo]() | [Homepage](https://www.bllossom.ai/) | [Github](https://github.com/MLP-Lab/Bllossom) | [Colab-tutorial](https://colab.research.google.com/drive/1fBOzUVZ6NRKk_ugeoTbAOokWKqSN47IG?usp=sharing) |

[Latest Update] 
- 2024.05.08 Vocab Expansion model update

The Bllossom language model is a Korean-English bilingual language model based on the open-source LLama3. It enhances the connection of knowledge between Korean and English. It has the following features:

* **Knowledge Linking**: Linking Korean and English knowledge through additional training
* **Vocabulary Expansion**: Expansion of Korean vocabulary to enhance Korean expressiveness.
* **Instruction Tuning**: Tuning using custom-made instruction following data specialized for Korean language and Korean culture
* **Human Feedback**: DPO has been applied
* **Vision-Language Alignment**: Aligning the vision transformer with this language model 

**This model developed by [MLPLab at Seoultech](http://mlp.seoultech.ac.kr), [Teddysum](http://teddysum.ai/) and [Yonsei Univ](https://sites.google.com/view/hansaemkim/hansaem-kim)**

## Demo Video

<div style="display: flex; justify-content: space-between;">
  <!-- 첫 번째 컬럼 -->
  <div style="width: 49%;">
    <a>
      <img src="https://github.com/lhsstn/lhsstn/blob/main/x-llava_dem.gif?raw=true" style="width: 100%; height: auto;">
    </a>
    <p style="text-align: center;">Bllossom-V Demo</p>
  </div>

  <!-- 두 번째 컬럼 (필요하다면) -->
  <div style="width: 49%;">
    <a>
      <img src="https://github.com/lhsstn/lhsstn/blob/main/bllossom_demo_kakao.gif?raw=true" style="width: 70%; height: auto;">
    </a>
    <p style="text-align: center;">Bllossom Demo(Kakao)ㅤㅤㅤㅤㅤㅤㅤㅤ</p>
  </div>
</div>



## NEWS
* [2024/04] We released Bllossom v2.0, based on llama-3
* [2023/12] We released Bllossom-Vision v1.0, based on Bllossom
* [2023/08] We released Bllossom v1.0, based on llama-2. 
* [2023/07] We released Bllossom v0.7, based on polyglot-ko.


```bash
저희 서울과기대 MLP연구실에서 한국어-영어 이중 언어모델인 Bllossom을 공개했습니다!
 - LLama3-8B 기반의 경량화된 사이즈 
 - 한국어-영어 지식연결을 통한 한국어 지식 강화
 - 한국어 어휘추가
 - 한국어 문화, 언어를 고려한 자체제작 데이터 기반 미세조정
 - 강화학습 (DPO)
 - 시각-언어 모델확장

1. Bllossom은 서울과기대, 테디썸, 연세대 언어자원 연구실의 언어학자와 협업해 만든 실용주의기반 언어모델입니다! 앞으로 지속적인 업데이트를 통해 관리하겠습니다 많이 활용해주세요 🙂
2. Bllossom70B모델, 어휘확장모델, 시각-언어모델은 추후 공개할 예정입니다. (궁금하신분은 개별 연락주세요, GPU만 지원해주시면 무료로 드립니다!)
3. Bllossom은 NAACL2024, LREC-COLING2024 (구두) 발표로 채택되었습니다.
4. 좋은 언어모델 계속 업데이트 하겠습니다!! 한국어 강화를위해 공동 연구하실분 언제든 환영합니다!!
```


## Example code

### Colab Tutorial
 - [Inference-Code-Link](https://colab.research.google.com/drive/1fBOzUVZ6NRKk_ugeoTbAOokWKqSN47IG?usp=sharing)

### Install Dependencies
```bash
pip install torch transformers==4.40.0 accelerate
```

### Python code with Pipeline
```python
import transformers
import torch

model_id = "MLP-KTLim/llama3-Bllossom"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()

PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
instruction = "서울과학기술대학교 MLP연구실에 대해 소개해줘"

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty = 1.1
)

print(outputs[0]["generated_text"][len(prompt):])

# 서울과학기술대학교 MLP연구실은 멀티모달 자연어처리 연구를 하고 있습니다. 구성원은 임경태 교수와 김민준, 김상민, 최창수, 원인호, 유한결, 임현석, 송승우, 육정훈, 신동재 학생이 있습니다.
```

### Python code with AutoModel
```python

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'MLP-KTLim/llama3-Bllossom'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
instruction = "서울과학기술대학교 MLP연구실에 대해 소개해줘"

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty = 1.1
)

print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))
# 서울과학기술대학교 MLP연구실은 멀티모달 자연어처리 연구를 하고 있습니다. 구성원은 임경태 교수와 김민준, 김상민, 최창수, 원인호, 유한결, 임현석, 송승우, 육정훈, 신동재 학생이 있습니다.
```



## Citation
**Language Model**
```text
@misc{bllossom,
  author = {ChangSu Choi, Yongbin Jeong, Seoyoon Park, InHo Won, HyeonSeok Lim, SangMin Kim, Yejee Kang, Chanhyuk Yoon, Jaewan Park, Yiseul Lee, HyeJin Lee, Younggyun Hahm, Hansaem Kim, KyungTae Lim},
  title = {Optimizing Language Augmentation for Multilingual Large Language Models: A Case Study on Korean},
  year = {2024},
  journal = {LREC-COLING 2024},
  paperLink = {\url{https://arxiv.org/pdf/2403.10882}},
 },
}
```

**Vision-Language Model**
```text
@misc{bllossom-V,
  author = {Dongjae Shin, Hyunseok Lim, Inho Won, Changsu Choi, Minjun Kim, Seungwoo Song, Hangyeol Yoo, Sangmin Kim, Kyungtae Lim},
  title = {X-LLaVA: Optimizing Bilingual Large Vision-Language Alignment},
  year = {2024},
  publisher = {GitHub},
  journal = {NAACL 2024 findings},
  paperLink = {\url{https://arxiv.org/pdf/2403.11399}},
 },
}
```

## Contact
 - 임경태(KyungTae Lim), Professor at Seoultech. `ktlim@seoultech.ac.kr`
 - 함영균(Younggyun Hahm), CEO of Teddysum. `hahmyg@teddysum.ai`
 - 김한샘(Hansaem Kim), Professor at Yonsei. `khss@yonsei.ac.kr`

## Contributor
 - 최창수(Chansu Choi), choics2623@seoultech.ac.kr
 - 김상민(Sangmin Kim), sangmin9708@naver.com
 - 원인호(Inho Won), wih1226@seoultech.ac.kr
 - 김민준(Minjun Kim), mjkmain@seoultech.ac.kr 
 - 송승우(Seungwoo Song), sswoo@seoultech.ac.kr
 - 신동재(Dongjae Shin), dylan1998@seoultech.ac.kr
 - 임현석(Hyeonseok Lim), gustjrantk@seoultech.ac.kr
 - 육정훈(Jeonghun Yuk), usually670@gmail.com
 - 유한결(Hangyeol Yoo), 21102372@seoultech.ac.kr
 - 송서현(Seohyun Song), alexalex225225@gmail.com
