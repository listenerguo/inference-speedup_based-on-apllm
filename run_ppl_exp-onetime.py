import torch
import torch.nn as nn
from any_precision.modules import APForCausalLM, AnyPrecisionForCausalLM
from transformers import AutoModelForCausalLM
from any_precision.quantization.datautils import get_dataset_from_disk
import time
import os

def get_wikitext2(model, dataset_path):
    testdata = get_dataset_from_disk(dataset_path, 'test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    testenc = tokenizer("\n\n".join(testdata), return_tensors="pt")
    # torch.save(testenc)
    return testenc


@torch.no_grad()
def opt_eval(model, testenc, dev, seqlen=2048):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


def main():
    """
        用于给 原始模型进行 PPL 计算
    """
    # # 实验结果
    # qbit_model_dir = "anyprec-(opt-1.3b)-w8_orig4-gc1-wikitext2_s100_blk128"    # 执行 w = lut(w') 的 推理过程 对应的 模型
    qbit_model_dir = "ap-(opt-1.3b)-w8_orig4-gc128-c4_s100_blk256_0113_l2h"   # 执行 w = s(w'-z) 的 推理过程 对应的 模型
    dataset_path = "wikitext2"

    quant_bit = 6

    model_name = os.path.basename(qbit_model_dir.rstrip('/'))
    testenc = get_wikitext2(qbit_model_dir, dataset_path)

    model_type = model_name.split("-")[0]
    print(model_type)

    if model_type == "anyprec":
        # print()
        model = AnyPrecisionForCausalLM.from_quantized(qbit_model_dir)
    elif model_type == "ap":
        model = APForCausalLM.from_quantized(qbit_model_dir)
    else:
        print("error")
        return 0

    model = model.half()
    model.eval()
    model.set_precision(quant_bit)  # 设置模型推理精度
    print(f'ppl calculation of Any-precision_my:')
    tb_qbit = time.time()
    opt_eval(model.model, testenc, "cuda:0")
    tf_qbit = time.time()

    # 释放模型显存
    del model
    torch.cuda.empty_cache()

    print(f'{model_name}量化后-{quant_bit}bit-模型  执行PPL计算所用时间：{tf_qbit - tb_qbit}')


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()