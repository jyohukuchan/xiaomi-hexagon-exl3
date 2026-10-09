"""Capture layer-zero CUDA tensors without advancing or quantizing the KV cache."""
import argparse
import json
from pathlib import Path
import torch
from exllamav3 import Config, Model

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("directory", type=Path)
parser.add_argument("tokens", nargs="+", type=int)
parser.add_argument("--all-layers", action="store_true")
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
torch.cuda.set_device(0)
model = Model.from_config(Config.from_directory(args.model))
model.load(progressbar=True)

def save(name, value):
    if not isinstance(value, torch.Tensor):
        return
    value = value.detach().float().cpu().contiguous()
    value.numpy().tofile(args.directory / (name + ".bin"))
    (args.directory / (name + ".json")).write_text(json.dumps({"shape": list(value.shape)}))
    print("TRACE", name, list(value.shape), flush=True)

with torch.inference_mode():
    embedding = model.modules[0].embedding
    embedding.weight = torch.nn.Parameter(model.modules[-1].inner.get_weight_tensor().T.contiguous().to(embedding.weight.device), requires_grad=False)
    for module in model:
        if not (".layers.0." in module.key or (args.all_layers and ".layers." in module.key)):
            continue
        original = module.forward
        def wrapped(x, params, *a, original=original, key=module.key, **kw):
            save(key + ".input", x)
            result = original(x, params, *a, **kw)
            save(key + ".output", result)
            return result
        module.forward = wrapped
    params = {"attn_mode": "flash_attn_nc", "export_state_layers": [0]}
    x = model.modules[0].forward(torch.tensor([args.tokens], device=embedding.weight.device), params)
    save("embedding", x)
    if args.all_layers:
        x = model.forward(torch.tensor([args.tokens], device="cuda:0"), params)
        save("logits", x)
    else:
        x = model.modules[1].prepare_for_device(x, params)
        x = model.modules[1].forward(x, params)
        save("layer0", x)
