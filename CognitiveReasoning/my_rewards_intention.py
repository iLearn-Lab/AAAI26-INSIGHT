
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import re
from typing import List

import torch
import torch.nn.functional as F
import math
from torch import distributed as dist
from transformers import AutoTokenizer, AutoModel
import deepspeed

from swift.plugin import ORM
from swift.utils import get_logger


logger = get_logger(__name__)



def clean_text(text: str) -> str:

    return re.sub(r'[<>/]', ' ', text)


def english_soft_score(text: str) -> float:

    cleaned = clean_text(text)
    tokens_all = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", cleaned)
    if not tokens_all:
        return 0.0
    english_tokens = [t for t in tokens_all if re.fullmatch(r"[A-Za-z]+", t)]
    if len(english_tokens) != len(tokens_all):
        return 0.0
    return 1.0



def list_damerau_distance(seq1, seq2):

    m, n = len(seq1), len(seq2)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        d[i][0] = i
    for j in range(1, n + 1):
        d[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost
            )
            if (
                i > 1 and j > 1
                and seq1[i - 1] == seq2[j - 2]
                and seq1[i - 2] == seq2[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[m][n]




_INT_PREFIX_RE = re.compile(
    r"^\s*(?:The\s+most\s+likely\s+intention\s+of\s+the\s+task\s+is|Intention|Intent)\s*[,:-]*\s*",
    flags=re.I
)


def _clean_intention(text: str) -> str:

    cleaned = _INT_PREFIX_RE.sub("", text or "").strip()
    return cleaned if cleaned else " "


class ActionIntentReward(ORM):


    def __init__(
        self,
        target_pairs: int = 20,
        model_path: str = '.../all-MiniLM-L6-v2',
        use_half: bool = True,
        max_length: int = 256,
    ):
        super().__init__()
        self.T = target_pairs
        self.max_length = max_length
        self.use_half = use_half and torch.cuda.is_available()


        self.beta_sim = 0.8
        self.gamma = 40.0


        self.pattern_tags = re.compile(
            r'^.*?<think>.*?</think>.*?<intention>.*?</intention>.*?<answer>.*?</answer>.*$',
            flags=re.DOTALL
        )

        self.pattern_pair = re.compile(r'^[a-z]+ [a-z]+$', flags=re.I)


        self.rank = dist.get_rank() if dist.is_initialized() else 0

        local_rank_env = os.environ.get("LOCAL_RANK", None)
        if local_rank_env is not None:
            try:
                local_rank = int(local_rank_env)
            except ValueError:
                local_rank = self.rank
        else:
            local_rank = self.rank

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
        logger.info(f"[rank {self.rank}] using device {self.device}")


        torch_dtype = torch.float16 if self.use_half else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )
        self.encoder = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch_dtype,
        )
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.encoder.to(self.device)

        w = self.encoder.embeddings.word_embeddings.weight
        shape = tuple(w.shape)
        if shape[0] == 0:
            logger.warning(f"[rank {self.rank}] local shard rows=0")
        else:
            logger.info(f"[rank {self.rank}] embedding shape={shape}")
        self.hidden = shape[1] if len(shape) == 2 else w.shape[-1]

        self.use_zero_gather = False

    def _embed(self, texts: List[str]) -> torch.Tensor:

        ctx = (
            deepspeed.zero.GatheredParameters(list(self.encoder.parameters()), modifier_rank=None)
            if self.use_zero_gather else nullcontext()
        )
        with ctx:
            with torch.inference_mode():
                batch = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                hidden = self.encoder(**batch).last_hidden_state  # (B, L, D)
                mask = batch["attention_mask"].unsqueeze(-1)      # (B, L, 1)
                sent = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                sent = F.normalize(sent, p=2, dim=1)  # (B, D)
                return sent


    def _sim_to_score(self, sim: float) -> float:

        raw = 1.0 / (1.0 + math.exp(-self.gamma * (sim - self.beta_sim)))
        max_raw = 1.0 / (1.0 + math.exp(-self.gamma * (1.0 - self.beta_sim)))
        score = raw / max_raw
        return min(score, 1.0)

    def score_len(self, n_pairs: int) -> float:

        return 1.0 if n_pairs >= self.T else 0.0

    def score_cont(self, gen_pairs: list, gt_pairs: list) -> float:


        max_len = 20


        gt = gt_pairs[:max_len]
        gen = gen_pairs[:max_len]


        raw_dist = list_damerau_distance(gt, gen)


        actual_len = max(len(gt_pairs), len(gen_pairs))
        if actual_len < max_len:
            raw_dist += (max_len - actual_len)


        rel_err = raw_dist / max_len
        score = (1.0 - rel_err) ** 1


        return float(max(0.0, min(1.0, score)))

    def score_tags(self, text: str) -> float:

        return 1.0 if self.pattern_tags.match(text) else 0.0

    def score_lang(self, text: str) -> float:
        return english_soft_score(text)

    def _valid_pair(self, token: str) -> bool:
        return bool(self.pattern_pair.fullmatch(token.strip()))


    def _extract_reference_assistant(self, msg: list) -> str:

        for turn in reversed(msg):
            if turn.get("role") == "assistant" and "<answer>" in turn.get("content", ""):
                return turn["content"]
        return ""


    def __call__(self, completions: list, messages: list, **kwargs):

        rewards = []


        gt_from_loader = kwargs.get("gt_answer", None)     
        gt_int_from_loader = kwargs.get("gt_intention", None) 


        gt_int_texts = []
        gen_int_texts = []
        s_len_list, s_cont_list, s_fmt_list, s_lang_list = [], [], [], []
        gt_pairs_list, gen_pairs_list = [], []

        for idx, (pred_full, msg) in enumerate(zip(completions, messages)):

            if gt_from_loader is not None:
                gt_content = gt_from_loader[idx]
            else:
                gt_content = self._extract_reference_assistant(msg)


            if gt_int_from_loader is not None:
                gt_int_text = gt_int_from_loader[idx]
            else:
                m_gt = re.search(r'<intention>(.*?)</intention>', gt_content, flags=re.DOTALL)
                gt_int_text = m_gt.group(1) if m_gt else ""


            gt_actions = (
                gt_content.split('<answer>', 1)[1].split('</answer>', 1)[0]
                if '<answer>' in gt_content else ""
            )
            gen_actions = (
                pred_full.split('<answer>', 1)[1].split('</answer>', 1)[0]
                if '<answer>' in pred_full else ""
            )


            gt_pairs  = [tok.lower().strip() for tok in gt_actions.split(',') if self._valid_pair(tok)]
            gen_pairs = [tok.lower().strip() for tok in gen_actions.split(',') if self._valid_pair(tok)]


            s_len  = self.score_len(len(gen_pairs))
            s_cont = self.score_cont(gen_pairs, gt_pairs)
            s_fmt  = self.score_tags(pred_full)
            s_lang = self.score_lang(pred_full)


            s_len_list.append(s_len)
            s_cont_list.append(s_cont)
            s_fmt_list.append(s_fmt)
            s_lang_list.append(s_lang)
            gt_pairs_list.append(gt_pairs)
            gen_pairs_list.append(gen_pairs)


            m_pred = re.search(r'<intention>(.*?)</intention>', pred_full, flags=re.DOTALL)
            gen_int_text = m_pred.group(1) if m_pred else ""

            gt_int_texts.append(_clean_intention(gt_int_text))
            gen_int_texts.append(_clean_intention(gen_int_text))

        batch_size = len(completions)

        if any(t.strip() for t in gt_int_texts) or any(t.strip() for t in gen_int_texts):
            emb_all = self._embed(gt_int_texts + gen_int_texts)  # (2B, D) GPU
            gt_embs = emb_all[:batch_size]
            gen_embs = emb_all[batch_size:]
            sims = F.cosine_similarity(gt_embs, gen_embs, dim=1)  # (B,)
            s_int_list = [self._sim_to_score(float(s.item())) for s in sims]
        else:
            s_int_list = [0.0] * batch_size

        for i in range(batch_size):
            s_len  = s_len_list[i]
            s_cont = s_cont_list[i]
            s_fmt  = s_fmt_list[i]
            s_lang = s_lang_list[i]
            s_int  = s_int_list[i]

            R = s_len * (0.85 * s_cont + 0.05 * s_int + 0.05 * s_lang + 0.05 * s_fmt)
            R = float(max(0.0, min(1.0, R)))
            rewards.append(R)

            logger.info(
                f"[Reward] len={s_len:.3f}, cont={s_cont:.3f}, int={s_int:.3f}, "
                f"fmt={s_fmt:.3f}, lang={s_lang:.3f}, R={R:.3f}"
            )

        return rewards
