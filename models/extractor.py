import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import *


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


class KPExtractor(nn.Module):

    def __init__(self, args, config, model):
        super().__init__()
        self.args = args
        self.config = config
        self.model = model
        self.loss_fct = NTXentLoss(args.temperature).cuda(args.gpu)
        self.doc_extractor = nn.Linear(config.d_model, config.d_model)
        self.kp_extractor = nn.Linear(config.d_model, config.d_model)


    def __get_doc_embeddings(self, last_hidden, input_ids, attention_mask=None, emb_type="cls"):
        if emb_type == "cls":
            return last_hidden[:,0,:]
        elif emb_type == "eos":
            eos_mask = input_ids.eq(self.config.eos_token_id)
            return last_hidden[eos_mask, :].view(last_hidden.size(0), -1, last_hidden.size(-1))[:, -1, :] 
        elif emb_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif emb_type == "max":
            return (last_hidden * attention_mask.unsqueeze(-1)).max(1)[0]
        else:
            raise ValueError("Supported document embeddings: cls, eos, avg, max.")


    def __get_kp_embeddings(self, last_hidden, candidate_kp_masks):
        if self.args.kp_pooler == "sum":
            cand_kp_embs = torch.bmm(candidate_kp_masks, last_hidden)
        else:
            cand_kp_embs = []
            for i in range(candidate_kp_masks.shape[0]):
                embs = []
                for j in range(candidate_kp_masks.shape[1]):
                    token_embeddings = last_hidden[i][candidate_kp_masks[i][j]==1]
                    if len(token_embeddings) == 0:
                        emb = token_embeddings.new_zeros((1, last_hidden.shape[-1]))
                    else:
                        if self.args.kp_pooler == "logsumexp":
                            emb = torch.logsumexp(token_embeddings, dim=0)
                        elif self.args.kp_pooler == "max":
                            emb, _ = torch.max(token_embeddings, dim=0)
                        elif self.args.kp_pooler == "avg":
                            emb = torch.mean(token_embeddings, dim=0)
                        elif self.args.kp_pooler == "rnn":
                            emb = token_embeddings
                        else:
                            raise ValueError("Supported pooling operation: sum, logsumexp, max, avg, rnn, switch.")
                    embs.append(emb)
                embs = torch.stack(embs, dim=0)
                cand_kp_embs.append(embs)
                cand_kp_embs = torch.stack(cand_kp_embs, dim=0)

        return cand_kp_embs


    def __compute_scores(
        self,
        doc_embs,
        last_hidden,
        candidate_kp_masks,
        pre_kp_labels,
    ):
        ext_logits, ext_labels = [], []
        ext_logits_per_pos_score = []
        N, D = doc_embs.size()

        cand_kp_embs = self.__get_kp_embeddings(last_hidden, candidate_kp_masks)    # Pooling
        cand_kp_embs = torch.tanh(self.kp_extractor(cand_kp_embs))
        pos_samples = torch.masked_select(cand_kp_embs, (pre_kp_labels == 1).unsqueeze(-1)).view(-1, D)
        neg_samples = torch.masked_select(cand_kp_embs, (pre_kp_labels == 0).unsqueeze(-1)).view(-1, D)
        pos_batch_idxes = torch.nonzero(pre_kp_labels == 1, as_tuple=True)[0]
        neg_batch_idxes = torch.nonzero(pre_kp_labels == 0, as_tuple=True)[0]

        for i in range(N):
            pos_embs = pos_samples[pos_batch_idxes == i]
            neg_embs = neg_samples[neg_batch_idxes == i]
            if len(pos_embs) > 0:
                if self.args.dist_fn == "inner":         # bigger the better
                    pos_scores = pos_embs @ doc_embs[i]
                    neg_scores = neg_embs @ doc_embs[i]
                elif self.args.dist_fn == "cosine":      # bigger the better (max==1)
                    pos_scores = F.normalize(pos_embs) @ F.normalize(doc_embs[i].unsqueeze(0)).squeeze(0)
                    neg_scores = F.normalize(neg_embs) @ F.normalize(doc_embs[i].unsqueeze(0)).squeeze(0)
                elif self.args.dist_fn == "euclidean":   # smaller the better -> inverse it
                    pos_scores = 1 / (1 + torch.cdist(pos_embs, doc_embs[i].unsqueeze(0), p=2).squeeze(-1))
                    neg_scores = 1 / (1 + torch.cdist(neg_embs, doc_embs[i].unsqueeze(0), p=2).squeeze(-1))
                elif self.args.dist_fn == "manhattan":   # smaller the better -> inverse it
                    pos_scores = 1 / (1 + torch.cdist(pos_embs, doc_embs[i].unsqueeze(0), p=1).squeeze(-1))
                    neg_scores = 1 / (1 + torch.cdist(neg_embs, doc_embs[i].unsqueeze(0), p=1).squeeze(-1))
                else:
                    raise ValueError("Supported distance functions: inner, cosine, euclidean, manhattan.")

                for pos_score in pos_scores:
                    ext_logits_per_pos_score.append(torch.cat([pos_score.unsqueeze(0), neg_scores]))

                ext_logits.append(torch.cat([pos_scores, neg_scores]))
                ext_labels.append(torch.tensor([1] * len(pos_scores) + [0] * len(neg_scores), dtype=torch.float32))

        return ext_logits, ext_labels, ext_logits_per_pos_score


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        candidate_kp_masks=None,
        pre_kp_labels=None,
        labels=None,
    ):
        # Pass through LM to obtain contextual embeddings & cross-entropy loss for absent kp generation
        out = self.model(input_ids, attention_mask, labels=labels)

        last_hidden = out.encoder_last_hidden_state
        doc_embs = self.__get_doc_embeddings(last_hidden, input_ids, attention_mask, emb_type=self.args.doc_pooler)
        doc_embs = torch.tanh(self.doc_extractor(doc_embs))

        ext_logits, ext_labels, ext_logits_per_pos_score = self.__compute_scores(doc_embs,
                                                                                 last_hidden,
                                                                                 candidate_kp_masks,
                                                                                 pre_kp_labels)
        # None of the example in batch may have zero positive samples
        if (pre_kp_labels == 1).sum() == 0:
            ext_loss = torch.tensor(0.).cuda(self.args.gpu)
            model_output = (None, None)
        else:
            ext_logits_per_pos_score = pad_fn(ext_logits_per_pos_score, padding=float("-inf"))
            ext_logits = pad_fn(ext_logits, padding=float("-inf"))
            ext_labels = pad_fn(ext_labels, padding=-100).cuda(self.args.gpu)
            ext_loss = self.loss_fct(ext_logits_per_pos_score)
            model_output = (ext_logits, ext_labels)
        
        loss = out.loss + self.args.gamma * ext_loss
        model_output = model_output + (ext_loss, out.loss, loss)

        return model_output


    def extract(
        self,
        input_ids=None,
        attention_mask=None,
        candidate_kp_masks=None,
        pre_kp_labels=None,
        labels=None,
        theta=None,
    ):
        out = self.model(input_ids, attention_mask)

        last_hidden = out.encoder_last_hidden_state
        doc_embs = self.__get_doc_embeddings(last_hidden, input_ids, attention_mask, emb_type=self.args.doc_pooler)
        doc_embs = torch.tanh(self.doc_extractor(doc_embs))
        N, D = doc_embs.size()

        cand_kp_embs = self.__get_kp_embeddings(last_hidden, candidate_kp_masks)
        cand_kp_embs = torch.tanh(self.kp_extractor(cand_kp_embs))

        ext_preds = []
        for i in range(N):
            nonzero_idxes = (cand_kp_embs[i] != 0).any(1)
            kp_embs = cand_kp_embs[i][nonzero_idxes]

            kp_masks = candidate_kp_masks[i][nonzero_idxes]
            kp_labels = []
            for mask in kp_masks:
                mask_idxes = mask.nonzero(as_tuple=True)[0] # get all ones of sequence mask
                end = 0
                for j in range(1, len(mask_idxes)):         # get only the first span
                    if mask_idxes[0] + j == mask_idxes[j]:  # (only need to extract once for each word)
                        end = j
                    else:
                        break
                # Concatenate semicolons after each label
                cat = torch.cat([input_ids[i][mask_idxes[0:end+1]].to(torch.float),
                                 torch.tensor(self.config.semicolon_token_id, dtype=torch.float).unsqueeze(0).cuda(self.args.gpu)])
                kp_labels.append(cat)
            kp_labels = pad_fn(kp_labels, padding=self.config.pad_token_id)
            
            if self.args.dist_fn == "inner":         # bigger the better
                scores = kp_embs @ doc_embs[i]
            elif self.args.dist_fn == "cosine":      # bigger the better (max==1)
                scores = F.normalize(kp_embs) @ F.normalize(doc_embs[i].unsqueeze(0)).squeeze(0)
            elif self.args.dist_fn == "euclidean":   # smaller the better -> inverse it
                scores = 1 / (1 + torch.cdist(kp_embs, doc_embs[i].unsqueeze(0), p=2).squeeze(-1))
            elif self.args.dist_fn == "manhattan":   # smaller the better -> inverse it
                scores = 1 / (1 + torch.cdist(kp_embs, doc_embs[i].unsqueeze(0), p=1).squeeze(-1))
            else:
                raise ValueError("Supported distance functions: inner, cosine, euclidean, manhattan.")
        
            _, pred_idxes_at_k = torch.topk(scores, k=5)
            pred_idxes_at_m = (scores > theta).nonzero(as_tuple=True)[0]
            pred_idxes_at_km = pred_idxes_at_k if len(pred_idxes_at_m) < 5 else pred_idxes_at_m
            ext_preds.append(kp_labels[pred_idxes_at_km].view(-1))

        ext_preds = pad_fn(ext_preds, padding=self.config.pad_token_id).to(torch.long)
        return ext_preds


    def generate(self, inputs=None, **kwargs):
        return self.model.generate(inputs, **kwargs)
