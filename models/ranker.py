import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import NTXentLoss


def pad_fn(lst, padding=0):
    if len(lst) == 0:
        return lst
    max_len = max([x.shape[-1] for x in lst])
    if len(lst[0].shape) == 1:
        for i, x in enumerate(lst):
            lst[i] = F.pad(x, (0, max_len-x.shape[-1]), "constant", padding)
        lst = torch.stack(lst, dim=0)
    elif len(lst[0].shape) == 2:
        for i, x in enumerate(lst):
            lst[i] = F.pad(x, (0, max_len-x.shape[-1]), "constant", padding)
        lst = torch.cat(lst, dim=0)  # (batch_size, max_len)
    return lst


def pad_fn2(lst, padding=0):
    if len(lst) == 0:
        return lst
    if len(lst[0].shape) == 2:
        max_len = max([x.shape[0] for x in lst])
        for i, x in enumerate(lst):
            lst[i] = F.pad(x, (0, 0, 0, max_len-x.shape[0]), "constant", padding)
        lst = torch.stack(lst, dim=0)
    elif len(lst[0].shape) == 3:
        max_len = max([x.shape[1] for x in lst])
        for i, x in enumerate(lst):
            lst[i] = F.pad(x, (0, 0, 0, max_len-x.shape[1], 0, 0), "constant", padding)
        lst = torch.cat(lst, dim=0)
    return lst


class KPRanker(nn.Module):

    def __init__(self, args, config, model, model2):
        super().__init__()
        self.args = args
        self.config = config
        self.model = model
        self.model2 = model2

        self.loss_fct = NTXentLoss(args.temperature).cuda(args.gpu)

        self.doc_extractor = nn.Linear(config.hidden_size, 768)
        self.kp_extractor = nn.Linear(config.hidden_size, 768)


    def __get_doc_embeddings(self, out, attention_mask, emb_type):
        if emb_type == "cls":
            return out[:,0,:]
        elif emb_type == "avg":
            return (out * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif emb_type == "max":
            return (out * attention_mask.unsqueeze(-1)).max(1)[0]
        else:
            raise ValueError("Supported poolers: cls, avg, max")


    def __get_kp_embeddings(self, last_hidden, emb_type):
        if emb_type == "cls":
            return last_hidden[:,:,0,:]
        else:
            raise ValueError("Supported poolers: cls")


    def __compute_scores(self, candidate_embs, doc_emb, cand_labels):
        logits, labels = [], []
        logits_per_pos_score = []
        N, D = doc_emb.size()

        pos_samples = torch.masked_select(candidate_embs, (cand_labels == 1).unsqueeze(-1)).view(-1, D)
        neg_samples = torch.masked_select(candidate_embs, (cand_labels == 0).unsqueeze(-1)).view(-1, D)
        pos_batch_idxes = torch.nonzero(cand_labels == 1, as_tuple=True)[0]
        neg_batch_idxes = torch.nonzero(cand_labels == 0, as_tuple=True)[0]

        for i in range(N):
            pos_embs = pos_samples[pos_batch_idxes == i]
            neg_embs = neg_samples[neg_batch_idxes == i]
            if len(pos_embs) > 0:
                if self.args.dist_fn == "inner":         # bigger the better
                    pos_scores = pos_embs @ doc_emb[i]
                    neg_scores = neg_embs @ doc_emb[i]
                elif self.args.dist_fn == "cosine":      # bigger the better (max==1)
                    pos_scores = F.normalize(pos_embs) @ F.normalize(doc_emb[i].unsqueeze(0)).squeeze(0)
                    neg_scores = F.normalize(neg_embs) @ F.normalize(doc_emb[i].unsqueeze(0)).squeeze(0)
                elif self.args.dist_fn == "euclidean":   # smaller the better -> inverse it
                    pos_scores = 1 / (1 + torch.cdist(pos_embs, doc_emb[i].unsqueeze(0), p=2).squeeze(-1))
                    neg_scores = 1 / (1 + torch.cdist(neg_embs, doc_emb[i].unsqueeze(0), p=2).squeeze(-1))
                elif self.args.dist_fn == "manhattan":   # smaller the better -> inverse it
                    pos_scores = 1 / (1 + torch.cdist(pos_embs, doc_emb[i].unsqueeze(0), p=1).squeeze(-1))
                    neg_scores = 1 / (1 + torch.cdist(neg_embs, doc_emb[i].unsqueeze(0), p=1).squeeze(-1))
                else:
                    raise ValueError("Supported distance functions: inner, cosine, euclidean, manhattan.")

                logits.append(torch.cat([pos_scores, neg_scores]))
                labels.append(torch.tensor([1] * len(pos_scores) + [0] * len(neg_scores), dtype=torch.float32))

                for pos_score in pos_scores:
                    logits_per_pos_score.append(torch.cat([pos_score.unsqueeze(0), neg_scores]))

        return logits, labels, logits_per_pos_score


    def forward(
        self,
        src_input_ids=None,
        candidate_ids=None,
        cand_labels=None,
    ):
        # Get document embedding
        attention_mask = src_input_ids != self.config.pad_token_id
        out = self.model(src_input_ids, attention_mask)[0]
        doc_emb = self.__get_doc_embeddings(out, attention_mask, emb_type=self.args.doc_pooler)
        doc_emb = torch.tanh(self.doc_extractor(doc_emb))

        # Get candidate keyphrase embeddings
        batch_size, candidate_num, candidate_maxlen = candidate_ids.size()
        candidate_ids = candidate_ids.view(-1, candidate_ids.size(-1))
        attention_mask = candidate_ids != self.config.pad_token_id
        out = self.model2(candidate_ids, attention_mask)[0]
        last_hidden = out.view(batch_size, candidate_num, candidate_maxlen, -1)
        candidate_embs = self.__get_kp_embeddings(last_hidden, emb_type=self.args.kp_pooler)
        candidate_embs = torch.tanh(self.kp_extractor(candidate_embs))

        # Compute logits & loss
        logits, labels, logits_per_pos_score = self.__compute_scores(candidate_embs, doc_emb, cand_labels)
        logits_per_pos_score = pad_fn(logits_per_pos_score, padding=float("-inf"))
        logits = pad_fn(logits, padding=float("-inf"))
        labels = pad_fn(labels, padding=-100).cuda(self.args.gpu)
        loss = self.loss_fct(logits_per_pos_score)
        model_output = (loss, logits, labels)

        return model_output


    def rank(
        self,
        src_input_ids=None,
        candidate_ids=None,
        cand_labels=None,
        theta=None,
    ):
        # Get document embedding
        attention_mask = src_input_ids != self.config.pad_token_id
        out = self.model(src_input_ids, attention_mask)[0]
        doc_emb = self.__get_doc_embeddings(out, attention_mask, emb_type=self.args.doc_pooler)
        doc_emb = torch.tanh(self.doc_extractor(doc_emb))

        # Get candidate keyphrase embeddings
        batch_size, candidate_num, candidate_maxlen = candidate_ids.size()
        _candidate_ids = candidate_ids.view(-1, candidate_ids.size(-1))
        attention_mask = _candidate_ids != self.config.pad_token_id
        out = self.model2(_candidate_ids, attention_mask)[0]
        last_hidden = out.view(batch_size, candidate_num, candidate_maxlen, -1)
        candidate_embs = self.__get_kp_embeddings(last_hidden, emb_type=self.args.kp_pooler)
        candidate_embs = torch.tanh(self.kp_extractor(candidate_embs))

        preds = []
        for i in range(batch_size):
            if self.args.dist_fn == "inner":         # bigger the better
                scores = candidate_embs[i] @ doc_emb[i]
            elif self.args.dist_fn == "cosine":      # bigger the better (max==1)
                scores = F.normalize(candidate_embs[i]) @ F.normalize(doc_emb[i].unsqueeze(0)).squeeze(0)
            elif self.args.dist_fn == "euclidean":   # smaller the better -> inverse it
                scores = 1 / (1 + torch.cdist(candidate_embs[i], doc_emb[i].unsqueeze(0), p=2).squeeze(-1))
            elif self.args.dist_fn == "manhattan":   # smaller the better -> inverse it
                scores = 1 / (1 + torch.cdist(candidate_embs[i], doc_emb[i].unsqueeze(0), p=1).squeeze(-1))
            else:
                raise ValueError("Supported distance functions: inner, cosine, euclidean, manhattan.")
            
            _, pred_idxes_at_k = torch.topk(scores, k=5)
            pred_idxes_at_m = (scores > theta).nonzero(as_tuple=True)[0]
            pred_idxes_at_km = pred_idxes_at_k if len(pred_idxes_at_m) < 5 else pred_idxes_at_m
            preds.append(candidate_ids[i][pred_idxes_at_km].view(-1))

        preds = pad_fn(preds, padding=self.config.pad_token_id).to(torch.long)
        return preds
