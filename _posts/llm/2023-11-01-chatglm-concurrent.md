---
title: 开源大语言模型 chatglm 简单的并发改造
categories: [深度学习]
description: 对 chatglm 开源版本进行修改，提升并发能力
keywords: 
- chatglm
- llm
- 大语言模型
- 并发
date: 2023-11-01
draft: false
---

### 总结
- 开源的 chatglm3-6b 只提供了连续生成的api，实际部署使用时，在只用了一个workers的情况下，如果有多人同时提问，必须要等到前一个回答全部结束后才会开始回答下一个问题，在用户端的感觉是等待时间过长，于是我参照chatglm3源码写了一个简单的并发api，显存要求更高一点，不过当有多人同时提问时，可以同时进行回答，回答速度会变慢，可以理解成是并发用户均分 token 生成速度。
- 方案为临时使用，后续使用其他的高性能推理框架替代

### 整体思路
修改generate函数，不是连续生成一整句，每次只做一次推理，使用fastapi写一个请求端服务，附带上下文进行多次请求，请求服务有多个workers时可以处理并发，不需要等一整句生成完成后再生成下一句

#### 实现过程
推理服务 api.py
```python
class Message(BaseModel):
    cache_id: str
    query: str
    history: List[List[str]|Any] = []
    model_name: str = "chatglm3-6b"
    temperature: float = 0.95
    top_p: float = 0.7
    max_length: int = 8192
    do_sample: bool = True


class CacheMessage(BaseModel):
    flag: str
    delta_text: str


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class ChatModel:
    def __init__(self, model_path: str = "/data/git_source/huggingface/THUDM/chatglm3-6b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = "cuda"
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
        self.model.eval()
        # self.redis = redis.Redis(host='localhost', port=6379, db=0, password="redispass")
        self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(InvalidScoreLogitsProcessor())
        self.stopping_criteria = StoppingCriteriaList()
        # 内存中保存还没回答完的句子数据
        self.cache = {}

    # 参考 chatglm 源码修改，每次生成只推理一次
    @torch.inference_mode()
    def generate(self, message: Message) -> CacheMessage:
            gen_kwargs = {"max_length": message.max_length, 
                        "do_sample": message.do_sample, 
                        "top_p": message.top_p,
                        "temperature": message.temperature, 
                        "logits_processor": self.logits_processor}
            kwargs = gen_kwargs
            # 是否是新的句子
            if message.cache_id in self.cache:
                msg = self.cache[message.cache_id]
                if msg["flag"] == "end":
                    del self.cache[message.cache_id]
                    return {"flag": msg["flag"], 
                            "delta_text": msg["delta_text"]}
                input_ids = msg["input_ids"]
                model_kwargs = self.cache[message.cache_id]["model_kwargs"]
            # 新句子生成一个唯一id
            else:
                inputs = self.tokenizer.build_chat_input(message.query, history=message.history, role="user")
                input_ids = inputs["input_ids"].to(self.device)
                model_kwargs = self.model.generation_config.update(**kwargs)
                model_kwargs["use_cache"] = self.model.generation_config.use_cache
                msg = {
                    "flag": "sending",
                    "input_ids": input_ids,
                    "model_kwargs": model_kwargs,
                    "input_ids_raw_len": input_ids.shape[1],
                    "previous_text": "",
                    "delta_text": "",
                    "create": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.cache[message.cache_id] = msg
            # 推理过程
            _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
            _, eos_token_id = self.model.generation_config.bos_token_id, self.model.generation_config.eos_token_id
            eos_token_id = [eos_token_id]
            eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
            logits_processor = self.model._get_logits_processor(
                generation_config=self.model.generation_config,
                input_ids_seq_length=input_ids_seq_length,
                encoder_input_ids=input_ids,
                prefix_allowed_tokens_fn=None,
                logits_processor=self.logits_processor,
            )

            stopping_criteria = self.model._get_stopping_criteria(
                generation_config=self.model.generation_config, stopping_criteria=self.stopping_criteria
            )
            logits_warper = self.model._get_logits_warper(self.model.generation_config)
            unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            if self.model.generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
            response = self.tokenizer.decode(input_ids.tolist()[0][self.cache[message.cache_id]["input_ids_raw_len"]:-1])
            self.cache[message.cache_id]["input_ids"] = input_ids
            if response:
                delta_text = response[len(self.cache[message.cache_id]["previous_text"]):]
                self.cache[message.cache_id]["delta_text"] = delta_text
                if response[-1] != "�":
                    self.cache[message.cache_id]["flag"] = "sending"
                    self.cache[message.cache_id]["previous_text"] = response
                else:
                    self.cache[message.cache_id]["flag"] = "hang"
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, None):
                self.cache[message.cache_id]["flag"] = "end"

            gc.collect()
            torch.cuda.empty_cache()
            return {"flag": self.cache[message.cache_id]["flag"], 
                    "delta_text": self.cache[message.cache_id]["delta_text"]}
```
请求推理的服务，chatglm.py
```python
    async def stream_chat(self, prompt: str, history: List[List[str]] = [], **kw):
        for k in self.chat_config:
            if k not in kw:
                kw[k] = self.chat_config[k]
        msg_history = []
        if len(history) > 0:
            for q, a in history:
                msg_history.append({"role": "user", "content": q})
                msg_history.append({"role": "assistant", "content": a})
        msg_history.append({"role": "user", "content": prompt})
        msg = {
            "cache_id": str(uuid.uuid4()),
            "query": prompt,
            "history": msg_history,
            **kw}
        headers = {'Content-Type': 'application/json'}
        history += [[]]
        # 多次请求推理服务，直到触发句子结束，句子结束后再次请求，会重新推理生成一遍
        while True:
            payload = json.dumps(msg)
            response = requests.post(f"http://{self.config['server_url']}/llm/generate", headers=headers, data=payload)
            if response.status_code == 200:
                resp = response.json()
                self.loginfo(f"raw response: delta_text {resp['delta_text']}")
                if resp["flag"] in ("sending", "end"):
                    r = resp["delta_text"]
                    history[-1] = [prompt, r]
                    answer_result = AnswerResult()
                    answer_result.history = history
                    answer_result.llm_output = {"answer": r}
                    yield answer_result
                    if resp["flag"] == "end":
                        break
            else:
                break
```
需要起两个服务，api.py 的服务只跑一个 workers（显存够大的话也可以跑多个），chatglm.py 按照并发要求跑多个 workers。

实际使用可以发现，当有多个问题同时提交时，后提交的不需要再等前一个回答完成才收到流式回复，而是会立刻开始收到回答。