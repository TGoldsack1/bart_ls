import logging

from fairseq.modules import (
  TransformerDecoder,
  MultiheadAttention,
)

from fairseq.models.transformer import TransformerEncoder


class GraphTransformerEncoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )
        self.args = args

        del self.layers
        self.layers = nn.ModuleList([self.build_graph_decoder_layer(args) for i in range(args.encoder_layers)])

          
    def build_graph_decoder_layer(self, cfg, no_encoder_attn=False):
        # layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)
        layer = GraphTransformerDecoderLayer(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class GraphTransformerDecoderLayer(TransformerEncoderLayer):
  
