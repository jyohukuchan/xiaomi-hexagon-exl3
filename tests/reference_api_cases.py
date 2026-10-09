"""Produce deterministic CUDA references for the public API regression cases."""
import argparse
import json
from pathlib import Path
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.generator.sampler import GreedySampler

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Output already exists")
cases = json.loads((Path(__file__).parent / "translation_cases.json").read_text(encoding="utf-8"))
torch.cuda.set_device(0)
config = Config.from_directory(args.model)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8192, max_batch_size=8)
tokenizer = Tokenizer.from_config(config)
model.load(progressbar=True)
with torch.inference_mode():
    embedding = model.modules[0].embedding
    embedding.weight = torch.nn.Parameter(model.modules[-1].inner.get_weight_tensor().T.contiguous().to(embedding.weight.device), requires_grad=False)
generator = Generator(model, cache, tokenizer)
for case in cases:
    case["user_prompt"] = "日语翻译成英语：" + case["text"]
    case["prompt"] = "<|im_start|>user\n" + case["user_prompt"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
responses = generator.generate(prompt=[c["prompt"] for c in cases], max_new_tokens=12,
    completion_only=True, encode_special_tokens=True, stop_conditions=config.eos_token_id_list, sampler=GreedySampler())
for case, response in zip(cases, responses, strict=True):
    case["expected"] = response
    print(case["id"], repr(response), flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({"max_tokens": 12, "tied_quantized_input": True, "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
