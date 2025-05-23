# modified from https://github.com/hahnyuan/ASVD4LLM
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from lm_eval.base import BaseLM
from lm_eval import evaluator
from typing import Optional, Literal
from dataset.loader import get_evaluation_dataloader
import logging

logger = logging.getLogger(__name__)

def setup_logger(log_file=None):
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if log_file:
        handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class EvalLM(BaseLM):
    def __init__(
        self,
        model,
        tokenizer,
        device: Literal["cuda:0", "cpu"] = "cuda:0",
        batch_size=1,
    ):
        super().__init__()

        # assert isinstance(device, str)
        assert isinstance(batch_size, int)

        # self.model = model.to(self.device)
        self.model = model
        self.model.eval()

        self._device = self.model.device 

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.seqlen = 2048

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :self.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False)


@torch.no_grad()
def evaluate_perplexity(model, dataset, limit, device: Optional[Literal["cuda", "cpu"]]):
    """
    dataset: input ids tensor of shape [batch, sequence length]
    """
    nsamples, seqlen = dataset.size()

    nlls = []

    for i in range(nsamples):
        if i == limit:
            break
        input_ids = dataset[i : i + 1, :-1].to(model.device)
        labels = dataset[i : i + 1, 1:].contiguous()
        logits = model(input_ids=input_ids)[0]
        shift_logits = logits[:, :, :]
        shift_labels = labels.to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    logger.info("PPL: %s", ppl.item())
    if device == "cuda":
        logger.info("Weight Memory: %s MiB\n", torch.cuda.memory_allocated()/1024/1024)
    return ppl.item()


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    model_name,
    tasks: Literal["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"],
    eval_ppl="",
    num_fewshot=0,
    limit=-1,
    batch_size=1,
    is_peft_model: Optional[bool] = False,
    device: Literal["cuda", "cpu"] = "cuda",
    log_file: Optional[str] = None
):
    """
    model: model name
    limit: number of test samples for debug, set to -1 is no limit
    tasks: str tasks are split by ,
    num_fewshot: Number of examples in few-shot context
    eval_ppl: str datasets are split by , such as 'wikitext2,ptb,c4'
    log_file: Path to log file for saving program output
    """
    # Update logger with new log_file if provided
    setup_logger(log_file)

    lm = EvalLM(model, tokenizer, batch_size=batch_size, device=device)
    results = {}
    if eval_ppl:
        for dataset in eval_ppl.split(","):
            logger.info("load dataset %s", dataset)
            # cache_testloader = f"cache/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
            # if os.path.exists(cache_testloader):
            #     testloader = torch.load(cache_testloader, weights_only=False)
            #     logger.info("load benchmark from %s", cache_testloader)
            # else:
                # testloader = get_evaluation_dataloader(dataset, tokenizer)
                # torch.save(testloader, cache_testloader)
            testloader = get_evaluation_dataloader(dataset, tokenizer)
            testenc = testloader.input_ids
            nsamples = testenc.numel() // lm.seqlen
            if is_peft_model:
                use_cache = lm.model.model.config.use_cache
                lm.model.model.config.use_cache = False
                lm.model.model.eval()
            else:
                use_cache = lm.model.config.use_cache
                lm.model.config.use_cache = False
                lm.model.eval()
            nlls = []

            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
                if is_peft_model:
                    outputs = lm.model.model.model(batch)
                    hidden_states = outputs[0]
                    logits = lm.model.model.lm_head(hidden_states)
                else:
                    outputs = lm.model.model(batch)
                    hidden_states = outputs[0]
                    logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][:, 1:].to(lm.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * lm.seqlen))
            logger.info("%s %s", dataset, ppl.item())
            if is_peft_model:
                lm.model.model.config.use_cache = use_cache
            else:
                lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
    if tasks == "longbench":
        from tools.eval_longbench import eval_longbench, full_longeval_datasets, small_longeval_datasets

        longbench_results = eval_longbench(model, tokenizer, model_name, datasets=full_longeval_datasets)
        results.update(longbench_results)
        tasks=""
    elif tasks == "small_longbench":
        from tools.eval_longbench import eval_longbench, full_longeval_datasets, small_longeval_datasets

        longbench_results = eval_longbench(model, tokenizer, model_name, datasets=small_longeval_datasets)
        results.update(longbench_results)
        tasks=""
    if tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=tasks.split(","),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
            no_cache=True,
        )
        t_results = t_results["results"]
        acc_list = [t_results[key]["acc"] for key in t_results.keys() if "acc" in t_results[key]]
        t_results["mean"] = sum(acc_list) / len(acc_list)
        results.update(t_results)
        logger.info("%s", results)
        logger.info("\n\n===== mean acc: %s =====\n\n", sum(acc_list)/len(acc_list))

    return results
