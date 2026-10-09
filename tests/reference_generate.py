"""Generate a deterministic upstream CUDA reference for the real EXL3 model."""

import argparse
from pathlib import Path
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.generator.sampler import GreedySampler

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("prompt", type=Path)
parser.add_argument("--tied-input", action="store_true")
args = parser.parse_args()
torch.cuda.set_device(0)
config = Config.from_directory(args.model)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
cache = Cache(model, max_num_tokens=8192, max_batch_size=1)
model.load(progressbar=True)
if args.tied_input:
    with torch.inference_mode():
        weight = model.modules[-1].inner.get_weight_tensor().T.contiguous()
        embedding = model.modules[0].embedding
        embedding.weight = torch.nn.Parameter(weight.to(embedding.weight.device), requires_grad=False)
    print("Using quantized head for input embeddings", flush=True)
generator = Generator(model, cache, tokenizer)
response = generator.generate(prompt=args.prompt.read_text(encoding="utf-8"), max_new_tokens=32,
                              completion_only=True, encode_special_tokens=True,
                              stop_conditions=config.eos_token_id_list, sampler=GreedySampler())
print("REFERENCE:", response, flush=True)
