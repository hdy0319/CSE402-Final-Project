# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyBERTConfig:
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 128,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


### model.py 수정 (MultiHeadSelfAttention 최적화)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TinyBERTConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # 하나의 선형 레이어로 Q, K, V를 동시에 생성
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        # QKV를 한 번에 계산 후 분할
        qkv = self.qkv(hidden_states)  # (batch, seq_len, 3 * hidden)
        query_layer, key_layer, value_layer = qkv.chunk(3, dim=-1)

        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        def reshape_to_heads(x):
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        query_layer = reshape_to_heads(query_layer)
        key_layer = reshape_to_heads(key_layer)
        value_layer = reshape_to_heads(value_layer)

        # 스케일드 닷 프로덕트 어텐션
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.num_heads * self.head_dim)

        out = self.out_proj(context_layer)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, config: TinyBERTConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        # Self-Attention 서브레이어
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output = self.attention_dropout(attn_output)
        attn_output = self.attention_norm(hidden_states + attn_output)

        # Feed-Forward 서브레이어
        intermediate_output = self.intermediate(attn_output)
        intermediate_output = self.intermediate_act_fn(intermediate_output)

        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_norm(attn_output + layer_output)
        return layer_output


class BERTModel(nn.Module):
    def __init__(self, config: TinyBERTConfig):
        super().__init__()
        self.config = config

        # Token Embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size + 1, config.hidden_size)
        # Position Embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # Token_type Embeddings (segment)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm & Dropout
        self.embeddings_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.embeddings_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer Blocks
        self.encoder_layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # input_ids: (batch, seq_len)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (batch, seq_len)

        # 임베딩 합산
        word_embeds = self.word_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + pos_embeds + type_embeds
        embeddings = self.embeddings_norm(embeddings)
        embeddings = self.embeddings_dropout(embeddings)

        # attention_mask을 4차원으로 변환: (batch, 1, 1, seq_len)
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # mask의 0 위치에 -10000.0을 더해서 어텐션에서 무시하게 함
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        hidden_states = embeddings
        for layer_module in self.encoder_layers:
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states  # (batch, seq_len, hidden_size)


class BERTForPreTraining(nn.Module):
    def __init__(self, config: TinyBERTConfig):
        super().__init__()
        self.config = config  # BERTModel의 config를 사용
        self.bert = BERTModel(config)
        self.mlm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        # MLM 예측을 위한 Linear(𝐡 → vocab)
        self.mlm_classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.mlm_classifier.weight = self.bert.word_embeddings.weight  # 가중치 공유

        # 가중치 초기화
        self.init_weights(config)

    def init_weights(self, config: TinyBERTConfig):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, mlm_labels=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask)  # (batch, seq_len, hidden)
        prediction_scores = self.mlm_dense(sequence_output)
        prediction_scores = self.activation(prediction_scores)
        prediction_scores = self.layer_norm(prediction_scores)
        prediction_scores = self.mlm_classifier(prediction_scores)  # (batch, seq_len, vocab_size)

        loss = None
        if mlm_labels is not None:
            # mlm_labels: (batch, seq_len), -100이 아닌 위치만 로스 계산
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # (batch * seq_len, vocab_size) vs (batch * seq_len,)
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        return prediction_scores, loss


class BERTForSequenceClassification(nn.Module):
    def __init__(self, config: TinyBERTConfig, num_labels: int):
        super().__init__()
        self.bert = BERTModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.init_weights(config)

    def init_weights(self, config: TinyBERTConfig):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask)  # (batch, seq_len, hidden)
        # [CLS] 토큰의 hidden state 사용 (첫 번째 토큰)
        cls_output = sequence_output[:, 0, :]  # (batch, hidden)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # (batch, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss