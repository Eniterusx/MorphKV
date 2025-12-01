from typing import Optional, Tuple, Union, List
import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging
from morphkv.morph_cache import MorphOffloadedCache

logger = logging.get_logger(__name__)

class GPT2AttentionMorph(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        self.max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((self.max_positions, self.max_positions), dtype=torch.bool)).view(
                1, 1, self.max_positions, self.max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = nn.Linear(2 * self.embed_dim, self.embed_dim) # GPT2 uses Conv1D usually, simplified here to maintain logic or assume Conv1D import available
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            # Using nn.Linear to mimic Conv1D behavior if needed, or assume Conv1D is imported from transformers.pytorch_utils
            # In context of the patch, we assume standard GPT2 structure availability. 
            # Note: The sample code uses Conv1D. We will assume standard attributes exist.
            from transformers.pytorch_utils import Conv1D
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            
        from transformers.pytorch_utils import Conv1D
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        # MorphKV Specific Init
        self.garbage = [True] * config.num_hidden_layers
        self.morph_type = ""
        self.WIN_SIZE = 1_000_000_000
        self.MAX_CAPACITY = 1_000_000_000

        if hasattr(config, 'morphkv') and config.morphkv:
            self.WIN_SIZE = int(config.morphkv['window_size'])
            self.MAX_CAPACITY = int(config.morphkv['max_capacity'])
            self.morph_type = config.morphkv['morph_type']
            self.evict_after = config.morphkv['evict_after']
            self.window_queries = [None] * self.config.num_hidden_layers

    def morphkv_mask(self, scores, past_key_value, key_heads, query_heads):
        # Determine fusion strategy (Sum or Max)
        if "max" in self.morph_type or self.morph_type=='max_fused': 
            sim_tokens = torch.full_like(scores[:,:,-(1+1):-1,:], -torch.inf) 
            # Exclude current token, select top-k based on max scores in the window
            init_mask_attn = sim_tokens[:,:,-1:].scatter_(
                -1,
                torch.topk(
                    nn.functional.softmax(scores[:, :, -(self.WIN_SIZE+1):-1, :-(self.WIN_SIZE+1)], dim=-1).max(dim=2, keepdim=True)[0], 
                    dim=-1, 
                    k=self.MAX_CAPACITY-self.WIN_SIZE
                ).indices,
                0.0
            )
        elif "sum" in self.morph_type or self.morph_type=='sum_fused': 
            sim_tokens = torch.full_like(scores[:,:,-(1+1):-1,:], -torch.inf) 
            # Exclude current token, select top-k based on sum scores in the window
            init_mask_attn = sim_tokens[:,:,-1:].scatter_(
                -1,
                torch.topk(
                    nn.functional.softmax(scores[:, :, -(self.WIN_SIZE+1):-1, :-(self.WIN_SIZE+1)], dim=-1).sum(dim=2, keepdim=True), 
                    dim=-1, 
                    k=self.MAX_CAPACITY-self.WIN_SIZE
                ).indices,
                0.0
            )
        
        # Ensure we always attend to the window size tokens and the current token
        init_mask_attn[:, :, -1, -(self.WIN_SIZE+1):] = 0.0  
        
        # GPT2 uses equal number of Q/K heads (no GQA), so we apply the same mask
        past_key_value.cleanup(init_mask_attn, init_mask_attn, self.layer_idx)
        
        return (init_mask_attn + scores[:,:,-1:,:]), init_mask_attn

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        query_cache: List = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        self.window_queries = query_cache

        # Logic adapted from GPT2Attention.forward
        if self.is_cross_attention:
            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # Reshape to (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.view(*query_states.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(*key_states.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(*value_states.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)

        query_heads = query_states.shape[1]
        key_heads = key_states.shape[1]

        if past_key_value is None and "layer_past" in kwargs:
            past_key_value = kwargs["layer_past"]

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
            )

        # MorphKV: Cache queries
        if hasattr(self.config, 'morphkv') and self.config.morphkv:
             query_states = past_key_value.update_win_queries(query_states[:,:,-(self.WIN_SIZE+1):,:], self.layer_idx)

        # Compute Attention Weights
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value_states.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # MorphKV Logic: Use only in generative phase (seq_len == 1) and when capacity is exceeded
        if hasattr(self.config, 'morphkv') and self.config.morphkv and \
           key_states.shape[2] >= (1 + self.MAX_CAPACITY) * self.evict_after:
            
            if hidden_states.shape[1] == 1:
                # Calculate mask and evict
                attn_weights, init_mask = self.morphkv_mask(attn_weights, past_key_value, key_heads, query_heads)
                
                # Cleanup garbage if needed
                if self.garbage[self.layer_idx]:
                    torch.cuda.empty_cache()
                    past_key_value.cleaned[self.layer_idx] = True
                    self.garbage[self.layer_idx] = False
            else:
                self.garbage[self.layer_idx] = True
        elif past_key_value is not None:
             # Just for profiling memory if not evicting yet
             past_key_value.cleanup(None, None, self.layer_idx, dummy=True)

        # Standard Causal Masking (if not cross attention and not already handled by morphkv entirely)
        if not self.is_cross_attention:
            query_length, key_length = query_states.size(-2), key_states.size(-2)
            if key_length > self.bias.shape[-1]:
                 causal_mask = torch.tril(torch.ones((key_length, key_length), device=self.bias.device, dtype=torch.bool)).view(1, 1, key_length, key_length)
                 causal_mask = causal_mask[:, :, key_length - query_length : key_length, :key_length]
            else:
                 causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            _causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + _causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        
        # Reshape back to (bsz, seq_len, embed_dim)
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, (key_states, value_states))
        if output_attentions:
            outputs += (attn_weights,)

        return outputs



def _update_causal_mask(attention_mask, input_tensor):
    if attention_mask is None:
        return None
    
    if attention_mask.dim() == 2:
        # Expand to (batch, 1, 1, seq_len)
        attention_mask = attention_mask[:, None, None, :]
    
    # Create additive mask: 1.0 -> 0.0, 0.0 -> min_dtype
    dtype = input_tensor.dtype
    min_dtype = torch.finfo(dtype).min
    # attention_mask = (1.0 - attention_mask) * min_dtype
    attention_mask = torch.where(attention_mask == 0, min_dtype, 0.0).to(dtype)
    return attention_mask

def gpt2_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Tuple[Tuple[torch.Tensor]], Cache]] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # MorphKV: Use MorphOffloadedCache
    return_legacy_cache = False
    if use_cache:
        if past_key_values is None:
            # Initialize MorphOffloadedCache instead of DynamicCache
            past_key_values = MorphOffloadedCache(self.config.num_hidden_layers)
            return_legacy_cache = False # Treat as legacy for return structure if needed by calling code
        elif not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = MorphOffloadedCache.from_legacy_cache(past_key_values, self.config.num_hidden_layers)
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated. Converting to MorphOffloadedCache."
            )

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # MorphKV: Clamp position_ids to avoid out of bounds for long sequences on GPT2
    if hasattr(self.config, "max_position_embeddings"):
        position_ids = position_ids.clamp(0, self.config.max_position_embeddings - 1)

    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

    # Attention mask setup (Standard GPT2 logic)
    if attention_mask is not None and attention_mask.ndim < 4:
        attention_mask = attention_mask.view(batch_size, -1)
    
    # Use local _update_causal_mask instead of calling it on self
    causal_mask = _update_causal_mask(attention_mask, inputs_embeds)

    if self.config.add_cross_attention and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_attention_mask = None

    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)
    output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    all_hidden_states = () if output_hidden_states else None
    presents = () if use_cache else None

    for i, block in enumerate(self.h):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                past_key_values,
                cache_position,
                causal_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
            )
        else:
            # Pass query_cache if using MorphKV
            query_cache = None
            if hasattr(self.config, 'morphkv') and self.config.morphkv and isinstance(past_key_values, MorphOffloadedCache):
                 query_cache = past_key_values.query_cache

            outputs = block(
                hidden_states,
                layer_past=past_key_values,
                cache_position=cache_position,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                query_cache=query_cache, 
                **kwargs,
            )

        hidden_states = outputs[0]
        if use_cache:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (outputs[2],)

        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.ln_f(hidden_states)
    hidden_states = hidden_states.view(output_shape)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)



    if use_cache:
        # Return DynamicCache if available, otherwise legacy tuple
        if isinstance(past_key_values, Cache):
            past_key_values = past_key_values
        else:
            past_key_values = presents

    if return_legacy_cache and past_key_values is not None:
         past_key_values = past_key_values.to_legacy_cache()

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )

def gpt2_block_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    query_cache: Optional[List] = None,
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=cache_position,
        query_cache=query_cache,
        **kwargs
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]
    # residual connection
    hidden_states = attn_output + residual



    if encoder_hidden_states is not None:
        if not hasattr(self, "crossattention"):
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            )
        residual = hidden_states
        hidden_states = self.ln_cross_attn(hidden_states)
        cross_attn_outputs = self.crossattention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = cross_attn_outputs[0]
        # residual connection
        hidden_states = residual + attn_output
        outputs = outputs + cross_attn_outputs[2:]

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward_hidden_states = self.mlp(hidden_states)
    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs