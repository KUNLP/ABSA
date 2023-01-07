import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os


class ABSAModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.hidden_size = config.hidden_size

        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config,
                                                 cache_dir=os.path.join(args.cache_dir, args.model_name_or_path))
        self.category_extractor = nn.Linear(self.hidden_size, args.aspect_size)
        self.loss_fnt = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_masks, category_targets=None):
        encoding = self.encoder(input_ids, attention_mask=attention_masks)[0]
        cls = encoding[:, 0]

        category_logits = self.category_extractor(cls)

        hardmax = F.one_hot(torch.argmax(category_logits, dim=-1), self.args.aspect_size)
        category_predicts = torch.logical_or((torch.sigmoid(category_logits) > 0.5), hardmax).to(torch.int32)

        if category_targets is not None:
            category_loss = self.loss_fnt(category_logits, category_targets)
            return category_loss

        return category_predicts

class ABSA_LAN(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.hidden_size = config.hidden_size
        self.label_emb_size = args.label_emb_size
        self.lan_hidden = args.lan_hidden
        self.entity_size = args.entity_size
        self.aspect_size = args.aspect_size
        self.category_size = args.category_size

        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config,
                                                 cache_dir=os.path.join(args.cache_dir, args.model_name_or_path))

        self.category_embedding = nn.Embedding(self.category_size, self.label_emb_size)
        self.category_indices = torch.tensor([i for i in range(self.category_size)])
        self.category_lan = LAN(self.label_emb_size, self.hidden_size, self.lan_hidden)
        self.category_extractor = nn.Linear(self.lan_hidden, 1)

        self.loss_fnt = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_masks, entity_targets=None):
        encoding = self.encoder(input_ids, attention_mask=attention_masks)[0]
        cls = encoding[:, 0]

        batch_size = cls.size()[0]
        category_indices = torch.tile(self.category_indices, (batch_size, 1)).to(cls.device)
        category_emb = self.category_embedding(category_indices)

        # output: [b, l, h], score: [b, e, l]
        output, score, weight = self.category_lan(category_emb, encoding, get_weight=True)
        category_logits = self.category_extractor(output)
        category_logits = category_logits.squeeze(dim=-1)

        hardmax = F.one_hot(torch.argmax(category_logits, dim=-1), self.category_size)
        category_predicts = torch.logical_or((torch.sigmoid(category_logits) > 0.5), hardmax).to(torch.int32)

        if entity_targets is not None:
            category_loss = self.loss_fnt(category_logits, entity_targets)
            return category_loss

        return category_predicts

class ABSA_LAN_MTL(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.hidden_size = config.hidden_size
        self.label_emb_size = args.label_emb_size
        self.lan_hidden = args.lan_hidden
        self.entity_size = args.entity_size
        self.aspect_size = args.aspect_size
        self.category_size = args.category_size

        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config,
                                                 cache_dir=os.path.join(args.cache_dir, args.model_name_or_path))
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                       cache_dir=os.path.join(args.cache_dir, args.model_name_or_path))

        self.entity_embedding = nn.Embedding(self.entity_size, self.label_emb_size)
        self.entity_indices = torch.tensor([i for i in range(self.entity_size)])
        self.entity_lan = LAN(self.label_emb_size, self.hidden_size, self.lan_hidden)
        self.entity_extractor = nn.Linear(self.lan_hidden, 1)

        self.aspect_embedding = nn.Embedding(self.aspect_size, self.label_emb_size)
        self.aspect_indices = torch.tensor([i for i in range(self.aspect_size)])
        self.aspect_lan = LAN(self.label_emb_size, self.hidden_size, self.lan_hidden)
        self.aspect_extractor = nn.Linear(self.lan_hidden, 1)

        self.category_indices = torch.tensor([i for i in range(self.category_size)])
        self.category_lan = LAN(self.label_emb_size * 2, self.hidden_size, self.lan_hidden)
        self.category_extractor = nn.Linear(self.lan_hidden, 1)

        self.loss_fnt = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_masks, pair_entity_id, pair_aspect_id, entity_targets=None,
                aspect_targets=None, category_targets=None):
        encoding = self.encoder(input_ids, attention_mask=attention_masks)[0]
        cls = encoding[:, 0]

        batch_size = cls.size()[0]

        entity_indices = torch.tile(self.entity_indices, (batch_size, 1)).to(cls.device)
        entity_emb = self.entity_embedding(entity_indices)
        # output: [b, l, h], score: [b, e, l]
        output, score, weight = self.entity_lan(entity_emb, encoding, get_weight=True)
        entity_logits = self.entity_extractor(output)
        entity_logits = entity_logits.squeeze(dim=-1)

        # entities = ["제품 전체", "본품", "패키지/구성품", "브랜드"]
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # print(tokens)
        # for en, w in zip(entities, weight[0]):
        #     w_to_li = enumerate(w.cpu().numpy().tolist())
        #     sort_w = sorted(w_to_li, key=lambda item: -item[1])
        #     tor_5 = sort_w[:5]
        #     print(en, [(item[0], tokens[item[0]], round(item[1], 3)) for item in tor_5])
        # print()


        hardmax = F.one_hot(torch.argmax(entity_logits, dim=-1), self.entity_size)
        entity_predicts = torch.logical_or((torch.sigmoid(entity_logits) > 0.5), hardmax).to(torch.int32)

        aspect_indices = torch.tile(self.aspect_indices, (batch_size, 1)).to(cls.device)
        aspect_emb = self.aspect_embedding(aspect_indices)
        # output: [b, l, h], score: [b, e, l]
        output, score, weight = self.aspect_lan(aspect_emb, encoding, get_weight=True)
        aspect_logits = self.aspect_extractor(output)
        aspect_logits = aspect_logits.squeeze(dim=-1)

        hardmax = F.one_hot(torch.argmax(aspect_logits, dim=-1), self.aspect_size)
        aspect_predicts = torch.logical_or((torch.sigmoid(aspect_logits) > 0.5), hardmax).to(torch.int32)

        pair_entity_id = torch.tensor(pair_entity_id).to(cls.device)
        pair_aspect_id = torch.tensor(pair_aspect_id).to(cls.device)
        pair_entity_emb = self.entity_embedding(pair_entity_id)
        pair_aspect_emb = self.aspect_embedding(pair_aspect_id)
        # [c, d]
        category_indices = torch.tile(self.category_indices, (batch_size, 1)).to(cls.device)
        category_emb = torch.cat([pair_entity_emb, pair_aspect_emb], dim=-1)[category_indices]
        output, score, weight = self.category_lan(category_emb, encoding, get_weight=True)
        category_logits = self.category_extractor(output)
        category_logits = category_logits.squeeze(dim=-1)

        hardmax = F.one_hot(torch.argmax(category_logits, dim=-1), self.category_size)
        category_predicts = torch.logical_or((torch.sigmoid(category_logits) > 0.5), hardmax).to(torch.int32)

        if entity_targets is not None:
            entity_loss = self.loss_fnt(entity_logits, entity_targets)
            aspect_loss = self.loss_fnt(aspect_logits, aspect_targets)
            category_loss = self.loss_fnt(category_logits, category_targets)

            return entity_loss, aspect_loss, category_loss

        return entity_predicts, aspect_predicts, category_predicts

class ABSA_LAN_MTL_v2(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.hidden_size = config.hidden_size
        self.label_emb_size = args.label_emb_size
        self.lan_hidden = args.lan_hidden
        self.entity_size = args.entity_size
        self.aspect_size = args.aspect_size
        self.category_size = args.category_size

        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config,
                                                 cache_dir=os.path.join(args.cache_dir, args.model_name_or_path))
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                       cache_dir=os.path.join(args.cache_dir, args.model_name_or_path))

        self.entity_embedding = nn.Embedding(self.entity_size, self.label_emb_size)
        self.entity_indices = torch.tensor([i for i in range(self.entity_size)])
        self.entity_lan = MatrixLAN(self.label_emb_size, self.hidden_size, self.lan_hidden)

        self.aspect_embedding = nn.Embedding(self.aspect_size, self.label_emb_size)
        self.aspect_indices = torch.tensor([i for i in range(self.aspect_size)])
        self.aspect_lan = MatrixLAN(self.label_emb_size, self.hidden_size, self.lan_hidden)

        self.category_indices = torch.tensor([i for i in range(self.category_size)])
        self.category_lan = MatrixLAN(self.label_emb_size * 2, self.hidden_size, self.lan_hidden)

        self.loss_fnt = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_masks, pair_entity_id, pair_aspect_id, entity_targets=None,
                aspect_targets=None, category_targets=None):
        encoding = self.encoder(input_ids, attention_mask=attention_masks)[0]
        cls = encoding[:, 0]

        batch_size = cls.size()[0]

        entity_indices = torch.tile(self.entity_indices, (batch_size, 1)).to(cls.device)
        entity_emb = self.entity_embedding(entity_indices)
        # output: [b, l, h], score: [b, e, l]
        score, weight = self.entity_lan(entity_emb, encoding, get_weight=True)
        entity_logits = score.squeeze(1)

        entities = ["제품 전체", "본품", "패키지/구성품", "브랜드"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        print("\t"+"\t".join(tokens))
        for en, w in zip(entities, weight[0]):
            w_to_li = w.cpu().numpy().tolist()
            print(en, "\t"+"\t".join([str(round(item, 3)) for item in w_to_li]))
        print()

        hardmax = F.one_hot(torch.argmax(entity_logits, dim=-1), self.entity_size)
        entity_predicts = torch.logical_or((torch.sigmoid(entity_logits) > 0.5), hardmax).to(torch.int32)

        aspect_indices = torch.tile(self.aspect_indices, (batch_size, 1)).to(cls.device)
        aspect_emb = self.aspect_embedding(aspect_indices)
        # output: [b, l, h], score: [b, e, l]
        score, weight = self.aspect_lan(aspect_emb, encoding, get_weight=True)
        aspect_logits = score.squeeze(1)

        hardmax = F.one_hot(torch.argmax(aspect_logits, dim=-1), self.aspect_size)
        aspect_predicts = torch.logical_or((torch.sigmoid(aspect_logits) > 0.5), hardmax).to(torch.int32)

        pair_entity_id = torch.tensor(pair_entity_id).to(cls.device)
        pair_aspect_id = torch.tensor(pair_aspect_id).to(cls.device)
        pair_entity_emb = self.entity_embedding(pair_entity_id)
        pair_aspect_emb = self.aspect_embedding(pair_aspect_id)
        # [c, d]
        category_indices = torch.tile(self.category_indices, (batch_size, 1)).to(cls.device)
        category_emb = torch.cat([pair_entity_emb, pair_aspect_emb], dim=-1)[category_indices]
        score, weight = self.category_lan(category_emb, encoding, get_weight=True)
        category_logits = score.squeeze(1)

        hardmax = F.one_hot(torch.argmax(category_logits, dim=-1), self.category_size)
        category_predicts = torch.logical_or((torch.sigmoid(category_logits) > 0.5), hardmax).to(torch.int32)

        if entity_targets is not None:
            entity_loss = self.loss_fnt(entity_logits, entity_targets)
            aspect_loss = self.loss_fnt(aspect_logits, aspect_targets)
            category_loss = self.loss_fnt(category_logits, category_targets)

            return entity_loss, aspect_loss, category_loss

        return entity_predicts, aspect_predicts, category_predicts

class LAN(nn.Module):
    def __init__(self, input_hidden, label_hidden, attention_hidden):
        super().__init__()
        self.attention_hidden = attention_hidden
        self.q_linear = nn.Linear(input_hidden, attention_hidden)
        self.k_linear = nn.Linear(label_hidden, attention_hidden)
        self.v_linear = nn.Linear(label_hidden, attention_hidden)
        self.drop_out = nn.Dropout(0.1)

    def forward(self, input_1, input_2, get_weight=False):
        q = self.q_linear(input_1)
        k = self.k_linear(input_2)
        v = self.v_linear(input_2)
        d = torch.sqrt(torch.tensor(self.attention_hidden))

        # [b, q, k]
        score = torch.matmul(q, k.transpose(1, 2)) / d
        weight = F.softmax(score, dim=-1)
        # [b, q, d]
        output = self.drop_out(torch.matmul(weight, v))

        if get_weight:
            return output, score, weight

        return output, score

class HLAN(nn.Module):
    def __init__(self, input_hidden, label_hidden, attention_hidden):
        super().__init__()
        self.attention_hidden = attention_hidden
        self.q_linear = nn.Linear(input_hidden, attention_hidden)
        self.k_linear = nn.Linear(label_hidden, attention_hidden)
        self.v_linear = nn.Linear(label_hidden, attention_hidden)

        self.q_linear_2 = nn.Linear(label_hidden, attention_hidden)
        self.k_linear_2 = nn.Linear(attention_hidden+input_hidden, attention_hidden)
        self.v_linear_2 = nn.Linear(attention_hidden+input_hidden, attention_hidden)

    def forward(self, input_1, input_2, get_weight=False):
        # input_1 : tokens, input_2 : label
        q = self.q_linear(input_1)
        k = self.k_linear(input_2)
        v = self.v_linear(input_2)
        d = torch.sqrt(torch.tensor(self.attention_hidden))

        # [b, t, l]
        score = torch.matmul(q, k.transpose(1, 2)) / d
        weight = F.softmax(score, dim=-1)
        # [b, t, d]
        intermediate = torch.matmul(weight, v)

        q_2 = self.q_linear_2(input_2)
        k_2 = self.k_linear_2(torch.cat([input_1, intermediate], dim=-1))
        v_2 = self.v_linear_2(torch.cat([input_1, intermediate], dim=-1))

        # [b, l, t]
        score = torch.matmul(q_2, k_2.transpose(1, 2)) / d
        weight = F.softmax(score, dim=-1)
        # [b, l, d]
        output = torch.matmul(weight, v_2)

        if get_weight:
            return output, score, weight

        return output, score


class MatrixLAN(nn.Module):
    def __init__(self, input_hidden, label_hidden, attention_hidden):
        super().__init__()
        self.attention_hidden = attention_hidden
        self.q_linear = nn.Linear(input_hidden, attention_hidden)
        self.k_linear = nn.Linear(label_hidden, attention_hidden)
        self.v1_linear = nn.Linear(label_hidden, attention_hidden)
        self.v2_linear = nn.Linear(input_hidden, attention_hidden)

        self.q_linear_2 = nn.Linear(attention_hidden * 2, attention_hidden)
        self.k_linear_2 = nn.Linear(input_hidden, attention_hidden)

        self.drop_out = nn.Dropout(0.1)

    def forward(self, input_1, input_2, get_weight=False):
        # input_1: label, input_2: tokens
        q = self.q_linear(input_1)
        k = self.k_linear(input_2)
        v1 = self.v1_linear(input_2)
        v2 = self.v2_linear(input_1)
        d = torch.sqrt(torch.tensor(self.attention_hidden))

        # [b, q, k]
        score = torch.matmul(q, k.transpose(1, 2)) / d
        b, q_len, k_len = score.size()
        flatten_score = score.view(b, -1)
        weight = F.softmax(flatten_score, dim=-1)
        weight = weight.view(score.size())

        # [b, q, d] -> [b, 1, d]
        v1 = self.drop_out(torch.matmul(weight, v1))
        v1 = v1.sum(1, keepdim=True)
        # [b, k, d] -> [b, 1, d]
        v2 = self.drop_out(torch.matmul(weight.transpose(1, 2), v2))
        v2 = v2.sum(1, keepdim=True)

        # [b, 1, 2d]
        intermediate = torch.cat([v1, v2], dim=-1)
        # [b, 1, d]
        q_2 = self.q_linear_2(intermediate)
        # [b, q, d]
        k_2 = self.k_linear_2(input_1)
        # [b, 1, q]
        score = torch.matmul(q_2, k_2.transpose(1, 2))

        if get_weight:
            return score, weight

        return score