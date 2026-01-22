from .helpers import dataloader
from tqdm import tqdm
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_CACHE"] = "/home/guoyd/.cache/huggingface/datasets"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from .helpers.utils import vprint, logprint, get_tokenizer_type, name_splitter, base_model_name_to_hf_repo_name
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..modules import AnyPrecisionForCausalLM,APForCausalLM

import json
import lm_eval
from lm_eval.models.huggingface import HFLM
import time
import torch.nn as nn

current_dir = os.path.dirname(os.path.realpath(__file__))


def fake_pack(parent_path, verbose=True):
    # Load from non-packed parent model to simulate quantization
    # WARNING: This is for PPL research only, and should not be used for any other purpose
    import re
    logprint(verbose, f"Simulating Any-Precision model from non-packed parent model at {parent_path}")

    if os.path.isdir('./cache/fake_packed'):
        for file in os.listdir('./cache/fake_packed'):
            if parent_path.split("/")[-1] in file:
                logprint(verbose, f"Faked packed model already exists for {parent_path.split('/')[-1]}. Skipping...")
                return

    # Check if D&S quantization is used
    dns = parent_path.split("/")[-1].startswith("dns")

    fields = name_splitter(parent_path)
    # get the field wrapped in ()
    for field in fields:
        if field.startswith('(') and field.endswith(')'):
            base_model_name = field[1:-1]
            break
    else:
        raise ValueError(f"Could not find base model name in {parent_path}")
    original_model_repo = base_model_name_to_hf_repo_name(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(original_model_repo)

    logprint(verbose, f"Loading original model from {original_model_repo}")
    # Load the model from the original model repo
    model = AutoModelForCausalLM.from_pretrained(original_model_repo, torch_dtype=torch.float16,
                                                 trust_remote_code=True)

    logprint(verbose, f"Loading quantized weights from {parent_path}")
    # Load the qweights
    files = os.listdir(parent_path + '/weights')
    layer_count = len(files)  # this should suffice
    qweights = [None] * layer_count
    for file in tqdm(files, desc="Loading qweights", disable=not verbose):
        # filenames should be 'l0.pt'
        l = int(re.match(r'l(\d+).pt', file).group(1))
        qweights[l] = torch.load(parent_path + '/weights/' + file, weights_only=False)

    logprint(verbose, f"Loading LUTs from {parent_path}")
    # get a list of directories in the model_path
    dirs = os.listdir(parent_path)
    dirs.remove('weights')
    if dns:
        dirs.remove('sparse')
    luts = {}
    # Only the LUT directories should remain
    for lut_dir in dirs:
        # example: lut_3
        bit = int(re.match(r'lut_(\d+)', lut_dir).group(1))
        for file in tqdm(os.listdir(parent_path + '/' + lut_dir), desc=f"Loading {bit}-bit LUTs",
                         disable=not verbose):
            # example: l0.pt
            l = int(re.match(r'l(\d+).pt', file).group(1))
            if bit not in luts:
                luts[bit] = [None] * layer_count
            luts[bit][l] = torch.load(parent_path + '/' + lut_dir + '/' + file, weights_only=False)

    # Load D&S sparse weights if they exist
    sparse_model_weights = []
    if dns:
        logprint(verbose, f"D&S quantization detected. Loading sparse weights...")
        for l in range(layer_count):
            sparse_weights = torch.load(parent_path + f'/sparse/l{l}.pt', weights_only=False)
            sparse_model_weights.append(sparse_weights)

    logprint(verbose, f"Replacing qweights with centroids from LUTs...")

    max_bit = max(luts.keys())

    for bit in luts:
        state_dict = model.state_dict()
        for l in tqdm(range(layer_count), desc=f"Replacing qweights with {bit}-bit centroids", ):
            qweight = qweights[l]
            lut = luts[bit][l]
            for module_name in qweight:
                full_param_name_suffix = f".{l}.{module_name}.weight"
                matching_keys = [key for key in state_dict.keys() if key.endswith(full_param_name_suffix)]
                assert len(matching_keys) == 1, f"Expected 1 matching key, got {len(matching_keys)}"
                matching_key = matching_keys[0]

                module_qweight = qweight[module_name]
                module_lut = lut[module_name]
                module_weights = []
                for row_idx in range(module_qweight.shape[0]):
                    row_weights = []
                    for group_idx in range(module_qweight.shape[1]):
                        # fetch weights from the LUT
                        group_weights = module_lut[row_idx][group_idx][
                            module_qweight[row_idx][group_idx] >> (max_bit - bit)]
                        row_weights.append(torch.from_numpy(group_weights))
                    # join the group weights
                    row_weights = torch.cat(row_weights, dim=0)
                    module_weights.append(row_weights)
                module_weights = torch.stack(module_weights)
                # Add the sparse weights if they exist
                if dns:
                    sparse_weights = sparse_model_weights[l][module_name]
                    # get the indices of the sparse weights
                    sparse_indices = sparse_weights.indices()
                    # replace the weights with the sparse weights
                    module_weights[sparse_indices[0], sparse_indices[1]] = sparse_weights.values()
                state_dict[matching_key] = module_weights

        save_path = f'./cache/fake_packed/fake_anyprec-p{bit}-{parent_path.split("/")[-1]}'
        os.makedirs(save_path, exist_ok=True)
        torch.save(state_dict, save_path + '/pytorch_model.bin')
        tokenizer.save_pretrained(save_path)
        model.config.save_pretrained(save_path)
        logprint(verbose, f"{bit}-bit model saved to {save_path}")


@torch.no_grad()
def auto_model_load(model_path, device='cuda', dtype=torch.float16, verbose=True):
    """
    Args:
        model_path: path of the model to evaluate
        device: the device to use for evaluation, either 'cuda' or 'cpu'
        dtype: the dtype to use for evaluation, either torch.float16 or torch.float32
        verbose: whether to print progress

    Returns:
        (tokenizer, model) tuple loaded from the given path, with the given device and dtype.
    """
    logprint(verbose, "Loading tokenizer and model...")

    if os.path.basename(model_path).startswith("anyprec-"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AnyPrecisionForCausalLM.from_quantized(model_path).to(device)
        model = APForCausalLM.from_quantized(model_path).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype,trust_remote_code=True).to(device)

    logprint(verbose, f"{model.__class__.__name__} model loaded to device: {model.device}")

    tokenizer_type = get_tokenizer_type(model_path)

    if tokenizer_type is None:
        logprint(verbose, f"Unknown tokenizer type for {model_path}. Cannot use cached input tokens.")

    return tokenizer_type, tokenizer, model


@torch.no_grad()
def evaluate_ppl(model, tokenizer, testcases, verbose=True, chunk_size=2048, tokenizer_type=None):
    """
    Args:
        model: model to evaluate
        tokenizer: tokenizer to use
        testcases: testcases names to evaluate on, passed on to dataloader.get_loaders
        verbose: whether to print progress
        chunk_size: the size of the chunks into which the test set is split
        tokenizer_type: set to llama, llama-2, or opt to use cached input tokens
                        for the corresponding test set

    Returns:
        A dictionary of perplexity scores, with keys being the testcases names and values being the perplexity scores.

    Note that the perplexity scores are calculated over non-overlapping chunks of the test set.
    """

    if isinstance(model, APForCausalLM):# AnyPrecisionForCausalLM
        is_anyprec = True
    else:
        is_anyprec = False

    model.eval()

    results = {}
    time_res = {}

    supported_bits = model.precisions if is_anyprec else [None]
    print('所支持的精度范围：',supported_bits)
    # supported_bits = [8]

    for bit in supported_bits:
        start_time = time.time()
        if is_anyprec:
            logprint(verbose, f"<<<< Setting model precision to {bit}-bit... >>>>")
            model.set_precision(bit)

        for testcase_name in testcases: # 例如 wikitext2
            vprint(verbose, f"---------------------- {testcase_name} ----------------------")

            # input_tokens = _load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose) 
            input_tokens = out_load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose)

            input_tokens.to(model.device)

            logprint(verbose, "Calculating perplexity...")

            seq_len = input_tokens.input_ids.size(1)
            nsamples = seq_len // chunk_size  # floor(seq_len / chunk_size)

            neg_log_likelihoods = []
            for i in tqdm(range(nsamples), disable=not verbose):
                begin_loc = i * chunk_size

                input_ids = input_tokens.input_ids[:, begin_loc:begin_loc + chunk_size]

                # add BOS token for Gemma-7B
                # https://github.com/huggingface/transformers/issues/29250
                if 'gemma' in model.config.architectures[0].lower():
                    # Mostly harmless to other models, but a slight drop in ppl is observed
                    # Hence, we only add the BOS token for Gemma models for now
                    input_ids[:, 0] = tokenizer.bos_token_id

                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                    neg_log_likelihood = outputs.loss
                    neg_log_likelihoods.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(neg_log_likelihoods).mean())
            logprint(verbose, f"Perplexity: {ppl.item()}")

            results[f"{testcase_name}:{bit}-bit"] = ppl.item()
        end_time = time.time()
        print(f"{bit}-bit model exe-ppl time: {end_time-start_time}")
        time_res[f"{testcase_name}:{bit}-bit"] = end_time - start_time
        if not is_anyprec:
            break

    return results, time_res


@torch.no_grad()
def my_ppl(model, tokenizer, testcases, verbose=True, seq_size=2048, tokenizer_type=None):
    """
    Args:
        model: model to evaluate
        tokenizer: tokenizer to use
        testcases: testcases names to evaluate on, passed on to dataloader.get_loaders_with_path
        verbose: whether to print progress
        chunk_size: the size of the chunks into which the test set is split
        tokenizer_type: set to llama, llama-2, or opt to use cached input tokens
                        for the corresponding test set

    Returns:
        A dictionary of perplexity scores, with keys being the testcases names and values being the perplexity scores.

    Note that the perplexity scores are calculated over non-overlapping chunks of the test set.
    """

    if isinstance(model, APForCausalLM):# AnyPrecisionForCausalLM
        is_anyprec = True
    else:
        is_anyprec = False

    model.eval()

    results = {}

    supported_bits = model.precisions if is_anyprec else [None]
    print('所支持的精度范围：',supported_bits)

    for bit in supported_bits:
        start_time = time.time()
        if is_anyprec:
            logprint(verbose, f"<<<< Setting model precision to {bit}-bit... >>>>")
            model.set_precision(bit)

        for testcase_name in testcases: # 例如 wikitext2
            vprint(verbose, f"---------------------- {testcase_name} ----------------------")

            # input_tokens = _load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose) 
            input_tokens = out_load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose)

            input_tokens.to(model.device)

            logprint(verbose, "Calculating perplexity...")

            
            ppl = my_ppl_opt_eval(model.model, input_tokens, model.device, seq_size)

            results[f"{testcase_name}:{bit}-bit"] = ppl
        end_time = time.time()
        print(f"{bit}-bit model exe-ppl time: {end_time-start_time}")
        if not is_anyprec:
            break

    return results

@torch.no_grad()
def my_ppl_opt_eval(model, testenc, dev, seqlen=2048):  # 仅适用于OPT ?
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
        batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(dev)
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
        # print(i)
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
        shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()

@torch.no_grad()
def run_lm_eval(tokenizer, model, tasks, verbose=True):
    """ Run lm-eval on the given model and tasks and return the results.

    Receives an already initialized hf model, and a list of task names.
    """
    if isinstance(model, APForCausalLM): # AnyPrecisionForCausalLM
        is_anyprec = True
    else:
        is_anyprec = False

    model.eval()

    results = {}

    supported_bits = model.precisions if is_anyprec else [None]

    for bit in supported_bits:
        if is_anyprec:
            logprint(verbose, f"<<<< Setting model precision to {bit}-bit... >>>>")
            model.set_precision(bit)

        model_lm = HFLM(pretrained=model, tokenizer=tokenizer)  # lm_eval.models.huggingface.HFLM
        eval_results = lm_eval.simple_evaluate(model=model_lm, tasks=tasks)

        if verbose:
            logprint(verbose, json.dumps(eval_results['results'], indent=4))

        for task in tasks:
            results[f"{task}:{bit}-bit"] = eval_results['results'][task]

        if not is_anyprec:
            break

    return results


def _load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose):
    """ Load input tokens from cache if available, otherwise load from dataloader and save to cache. """
    input_tokens_cache_path = f"{current_dir}/input_tokens_cache/dataloader-{tokenizer_type}-{testcase_name}-test.pt"
    if tokenizer_type and os.path.exists(input_tokens_cache_path):
        logprint(verbose, f"Loading cached input tokens from {input_tokens_cache_path}...")
        input_tokens = torch.load(input_tokens_cache_path, weights_only=False)
    else:
        logprint(verbose, "Loading test set...")

        raw_text = dataloader.get_loaders(testcase_name)    # 根据数据集名称，加载数据 # any_precision/evaluate/input_tokens_cache

        logprint(verbose, "Tokenizing test set...")

        input_tokens = tokenizer(raw_text, return_tensors='pt')
        # save input_tokens to cache
        if tokenizer_type:
            logprint(verbose, f"Caching input tokens to {input_tokens_cache_path}...")
            # we must create the directory if it doesn't exist
            os.makedirs(os.path.dirname(input_tokens_cache_path), exist_ok=True)
            torch.save(input_tokens, input_tokens_cache_path)

    return input_tokens


def out_load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose):
    """ Load input tokens from cache if available, otherwise load from dataloader and save to cache. """
    # testcase_name 支持 'wikitext2'  'c4'  'ptb'
    input_tokens_cache_path = f"{current_dir}/input_tokens_cache/dataloader-{tokenizer_type}-{testcase_name}-test.pt"
    if tokenizer_type and os.path.exists(input_tokens_cache_path):
        logprint(verbose, f"Loading cached input tokens from {input_tokens_cache_path}...")
        input_tokens = torch.load(input_tokens_cache_path, weights_only=False)
    else:
        testcase_path_0 = "/home/guoyd/Datasets/"
        logprint(verbose, "Loading test set...")

        testcase_path = testcase_path_0 + testcase_name
        raw_text = dataloader.get_loaders_with_path(testcase_path)    # 根据数据集名称，加载数据 # any_precision/evaluate/input_tokens_cache

        logprint(verbose, "Tokenizing test set...")

        input_tokens = tokenizer(raw_text, return_tensors='pt')
        # save input_tokens to cache
        if tokenizer_type:
            logprint(verbose, f"Caching input tokens to {input_tokens_cache_path}...")
            # we must create the directory if it doesn't exist
            os.makedirs(os.path.dirname(input_tokens_cache_path), exist_ok=True)
            torch.save(input_tokens, input_tokens_cache_path)
    return input_tokens