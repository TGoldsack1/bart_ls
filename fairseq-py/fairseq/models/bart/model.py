# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""
from typing import Optional

import logging

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.utils import safe_getattr
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.transformer.transformer_config import TransformerConfig

from .hub_interface import BARTHubInterface


logger = logging.getLogger(__name__)


# @register_model("bart")
# class BARTModel(TransformerModel):
#     __jit_unused_properties__ = ["supported_targets"]

#     @classmethod
#     def hub_models(cls):
#         return {
#             "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
#             "bart.large": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz",
#             "bart.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz",
#             "bart.large.cnn": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz",
#             "bart.large.xsum": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz",
#         }

#     def __init__(self, args, encoder, decoder):
#         super().__init__(args, encoder, decoder)

#         # We follow BERT's random weight initialization
#         self.apply(init_bert_params)

#         self.classification_heads = nn.ModuleDict()
#         if hasattr(self.encoder, "dictionary"):
#             self.eos: int = self.encoder.dictionary.eos()

#     @staticmethod
#     def add_args(parser):
#         super(BARTModel, BARTModel).add_args(parser)
#         parser.add_argument(
#             "--pooler-dropout",
#             type=float,
#             metavar="D",
#             help="dropout probability in the masked_lm pooler layers",
#         )
#         parser.add_argument(
#             "--pooler-activation-fn",
#             choices=utils.get_available_activation_fns(),
#             help="activation function to use for pooler layer",
#         )
#         parser.add_argument(
#             "--spectral-norm-classification-head",
#             action="store_true",
#             help="Apply spectral normalization on the classification head",
#         )

#         parser.add_argument(
#             '--restrict-position-embed',
#             action='store_true',
#             default=False,
#             help="do no extend the position embeddings"
#         )

#         parser.add_argument(
#             '--sliding-window', 
#             action='store_true', 
#             default=False, 
#             help="use sliding window attention as in longformer",
#         )
#         parser.add_argument(
#             '--block-attention', 
#             action='store_true', 
#             default=False, 
#             help="use block attention",
#         )
#         parser.add_argument(
#             '--top-down', 
#             action='store_true', 
#             default=False, 
#             help="use block attention",
#         )
        
#     @classmethod
#     def build_decoder(cls, args, tgt_dict, embed_tokens):
#         # use vanilla attention for now
#         args.use_xformers = False
#         return super().build_decoder(
#             TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
#         )

#     @classmethod
#     def build_encoder(cls, args, src_dict, embed_tokens):
#         # if args.sliding_window:
#         #     from fairseq.models.long_transformers.sliding_window import SWTransformerEncoder
#         #     args.attention_window = [512]
#         #     return SWTransformerEncoder(args, src_dict, embed_tokens)
#         if args.block_attention:
#             args.window_size = 1024
#             from fairseq.models.long_transformers.block import BlockTransformerEncoder
#             return BlockTransformerEncoder(args, src_dict, embed_tokens)
#         # if args.top_down:
#         #     args.window_size = 1024
#         #     args.encoder_n1 = 8
#         #     args.encoder_n2 = 2
#         #     args.encoder_n3 = 4
#         #     from fairseq.models.long_transformers.top_down import TopDownTransformerEncoder
#         #     return TopDownTransformerEncoder(args, src_dict, embed_tokens)    
#         return super().build_encoder(
#             TransformerConfig.from_namespace(args), src_dict, embed_tokens
#         )
        
#     def load_state_dict(
#         self,
#         state_dict,
#         strict=True,
#         model_cfg = None,
#         args = None,
#     ):  
#         # if self.args.top_down:
#         #     strict=False
#         return super().load_state_dict(state_dict, strict, model_cfg, args)

#     @property
#     def supported_targets(self):
#         return {"self"}

#     def forward(
#         self,
#         src_tokens,
#         src_lengths,
#         prev_output_tokens,
#         features_only: bool = False,
#         classification_head_name: Optional[str] = None,
#         token_embeddings: Optional[torch.Tensor] = None,
#         return_all_hiddens: bool = True,
#         alignment_layer: Optional[int] = None,
#         alignment_heads: Optional[int] = None,
#         masked_unfiltered: Optional[torch.Tensor] = None,
#     ):
#         if classification_head_name is not None:
#             features_only = True

#         encoder_out = self.encoder(
#             src_tokens,
#             src_lengths=src_lengths,
#             token_embeddings=token_embeddings,
#             return_all_hiddens=return_all_hiddens
#         )

#         x, extra = self.decoder(
#             prev_output_tokens,
#             encoder_out=encoder_out,
#             features_only=features_only,
#             alignment_layer=alignment_layer,
#             alignment_heads=alignment_heads,
#             src_lengths=src_lengths,
#             return_all_hiddens=return_all_hiddens,
#         )
#         eos: int = self.eos
#         if classification_head_name is not None:
#             sentence_representation = x[
#                 src_tokens.eq(eos), :
#             ].view(x.size(0), -1, x.size(-1))[:, -1, :]
#             for k, head in self.classification_heads.items():
#                 # for torch script only supports iteration
#                 if k == classification_head_name:
#                     x = head(sentence_representation)
#                     break
#         return x, extra

#     @classmethod
#     def from_pretrained(
#         cls,
#         model_name_or_path,
#         checkpoint_file="model.pt",
#         data_name_or_path=".",
#         bpe="gpt2",
#         sample_break_mode="eos",
#         **kwargs,
#     ):
#         from fairseq import hub_utils

#         x = hub_utils.from_pretrained(
#             model_name_or_path,
#             checkpoint_file,
#             data_name_or_path,
#             archive_map=cls.hub_models(),
#             bpe=bpe,
#             load_checkpoint_heads=True,
#             sample_break_mode=sample_break_mode,
#             **kwargs,
#         )
#         return BARTHubInterface(x["args"], x["task"], x["models"][0])

#     def register_classification_head(
#         self, name, num_classes=None, inner_dim=None, **kwargs
#     ):
#         """Register a classification head."""
#         logger.info("Registering classification head: {0}".format(name))
#         if name in self.classification_heads:
#             prev_num_classes = self.classification_heads[name].out_proj.out_features
#             prev_inner_dim = self.classification_heads[name].dense.out_features
#             if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
#                 logger.warning(
#                     're-registering head "{}" with num_classes {} (prev: {}) '
#                     "and inner_dim {} (prev: {})".format(
#                         name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
#                     )
#                 )
#         self.classification_heads[name] = BARTClassificationHead(
#             input_dim=self.args.encoder_embed_dim,
#             inner_dim=inner_dim or self.args.encoder_embed_dim,
#             num_classes=num_classes,
#             activation_fn=self.args.pooler_activation_fn,
#             pooler_dropout=self.args.pooler_dropout,
#             do_spectral_norm=getattr(
#                 self.args, "spectral_norm_classification_head", False
#             ),
#         )

#     def upgrade_state_dict_named(self, state_dict, name):
#         super().upgrade_state_dict_named(state_dict, name)

#         prefix = name + "." if name != "" else ""
#         current_head_names = (
#             []
#             if not hasattr(self, "classification_heads")
#             else self.classification_heads.keys()
#         )

#         # Handle new classification heads present in the state dict.
#         keys_to_delete = []
#         for k in state_dict.keys():
#             if not k.startswith(prefix + "classification_heads."):
#                 continue

#             head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
#             num_classes = state_dict[
#                 prefix + "classification_heads." + head_name + ".out_proj.weight"
#             ].size(0)
#             inner_dim = state_dict[
#                 prefix + "classification_heads." + head_name + ".dense.weight"
#             ].size(0)

#             if getattr(self.args, "load_checkpoint_heads", False):
#                 if head_name not in current_head_names:
#                     self.register_classification_head(head_name, num_classes, inner_dim)
#             else:
#                 if head_name not in current_head_names:
#                     logger.warning(
#                         "deleting classification head ({}) from checkpoint "
#                         "not present in current model: {}".format(head_name, k)
#                     )
#                     keys_to_delete.append(k)
#                 elif (
#                     num_classes
#                     != self.classification_heads[head_name].out_proj.out_features
#                     or inner_dim
#                     != self.classification_heads[head_name].dense.out_features
#                 ):
#                     logger.warning(
#                         "deleting classification head ({}) from checkpoint "
#                         "with different dimensions than current model: {}".format(
#                             head_name, k
#                         )
#                     )
#                     keys_to_delete.append(k)
#         for k in keys_to_delete:
#             del state_dict[k]

#         def truncate_emb(key):
#             if key in state_dict:
#                 state_dict[key] = state_dict[key][:-1, :]

#         # When finetuning on translation task, remove last row of
#         # embedding matrix that corresponds to mask_idx token.
#         loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
#         if (
#             loaded_dict_size == len(self.encoder.dictionary) + 1
#             and "<mask>" not in self.encoder.dictionary
#         ):
#             truncate_emb("encoder.embed_tokens.weight")
#             truncate_emb("decoder.embed_tokens.weight")
#             truncate_emb("encoder.output_projection.weight")
#             truncate_emb("decoder.output_projection.weight")

#         # When continued pretraining on new set of languages for mbart,
#         # add extra lang embeddings at the end of embed_tokens.
#         # Note: newly added languages are assumed to have been added at the end.
#         if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
#             self.encoder.dictionary
#         ):
#             logger.info(
#                 "Adding extra language embeddings not found in pretrained model for "
#                 "continued pretraining of MBART on new set of languages."
#             )
#             loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
#                 -1, :
#             ]

#             num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
#             embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

#             new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
#             nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
#             new_lang_embed_to_add = new_lang_embed_to_add.to(
#                 dtype=state_dict["encoder.embed_tokens.weight"].dtype,
#             )

#             state_dict["encoder.embed_tokens.weight"] = torch.cat(
#                 [
#                     state_dict["encoder.embed_tokens.weight"][
#                         : loaded_dict_size - 1, :
#                     ],
#                     new_lang_embed_to_add,
#                     loaded_mask_token_embedding.unsqueeze(0),
#                 ]
#             )
#             state_dict["decoder.embed_tokens.weight"] = torch.cat(
#                 [
#                     state_dict["decoder.embed_tokens.weight"][
#                         : loaded_dict_size - 1, :
#                     ],
#                     new_lang_embed_to_add,
#                     loaded_mask_token_embedding.unsqueeze(0),
#                 ]
#             )

#         # Copy any newly-added classification heads into the state dict
#         # with their current weights.
#         if hasattr(self, "classification_heads"):
#             cur_state = self.classification_heads.state_dict()
#             for k, v in cur_state.items():
#                 if prefix + "classification_heads." + k not in state_dict:
#                     logger.info("Overwriting " + prefix + "classification_heads." + k)
#                     state_dict[prefix + "classification_heads." + k] = v



from transformers import AutoTokenizer, AutoModel
import pickle
import dgl
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import re

class GATModel(nn.Module):
    def __init__(self, in_size, hid_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(dgl.nn.GATConv(in_size, hid_size, heads[0], activation=F.elu))
        self.gat_layers.append(dgl.nn.GATConv(hid_size*heads[0], hid_size, heads[1], residual=True, activation=F.elu))
        self.gat_layers.append(dgl.nn.GATConv(hid_size*heads[1], hid_size, heads[2], residual=True, activation=None))

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            
            h = layer(g, h)
            if i == 2:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
        return h

from enum import Enum


## CHECK ALL "SELF" references
class GraphEncoder():
    def __init__(self):
        dataset = "eLife"
        self.scibert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.cuid2embs = pickle.load(open("/home/acp20tg/bart_ls/resources/my_umls_concept_all_selected_definitions_embeddings.pkl", 'rb'))
        self.tuid2embs = pickle.load(open("/home/acp20tg/bart_ls/resources/my_semtype_definitions_embeddings.pkl", 'rb'))
        # self.id2graph = pickle.load(open("/home/acp20tg/bart_ls/resources/eLife_graphs.pkl", 'rb'))
        self.graphs = {
            "train": pickle.load(open(f"/home/acp20tg/bart_ls/resources/{dataset}_fs/train_graphs.pkl", 'rb')),
            "val": pickle.load(open(f"/home/acp20tg/bart_ls/resources/{dataset}_fs/val_graphs.pkl", 'rb')),
            "test": pickle.load(open(f"/home/acp20tg/bart_ls/resources/{dataset}_fs/test_graphs.pkl", 'rb'))
        }
        self.GM = GATModel(50, 1024, heads=[4,4,4])
        self.NodeType = Enum('NodeType', ['Document', "Section" 'Metadata', 'Concept', "Semtype"])

    def is_concept_node(self, node_id):
        return re.match(r'^[C][0-9]{7}', node_id)

    def is_semtype_node(self, node_id):
        return re.match(r'^[T][0-9]{4}', node_id)

    def get_initial_embeddings(self, aid, nodes, edges, device):
        ret_nodes, ret_embs = [], []
        has_title_edges = [e for e in edges if e[1] == "has_title"]
        titles = [r[2] for r in has_title_edges]
        # print(has_title_edges)
        for n in nodes:
            if self.is_concept_node(n):           ## Concept nodes ##
                # if n in self.cuid2embs:      # has defintion embedding
                emb = torch.tensor(self.cuid2embs[n]).to(device)
                #print("cui ", emb.shape)
                ret_embs.append(emb[0])
                ret_nodes.append(n)
            else:                            ## Non-concept nodes ##
                # Semantic type node
                if self.is_semtype_node(n):
                    emb = torch.tensor(self.tuid2embs[n]).to(device)
                    #print("tui ", emb.shape)
                    ret_embs.append(emb[0])
                    ret_nodes.append(n)
                else:                        # titles, keywords, and section text
                    if aid in n:
                        if "_Abs" in n:
                            title = "Abstract"
                        else:  
                            title_e = [e for e in has_title_edges if e[0] == n][0]
                            title = title_e[2]
                        if title.strip():
                            emb = self.get_sentence_embeddings([title], device, True)[0]
                            ret_embs.append(emb)
                            ret_nodes.append(n)
                    else:                    # titles and keywords
                        if n not in titles:
                            emb = self.get_sentence_embeddings([n], device, True)[0]
                            ret_embs.append(emb)
                            ret_nodes.append(n)                        
            
        return ret_nodes, ret_embs

    def get_node_type(self, node):
        if self.is_concept_node(node):
            return NodeType.Concept

        if self.is_semtype_node(node):
            return NodeType.Semtype

        if node.startswith("elife-") or node.startswith("journal."):
            if "_Abs" or "_Sec" in node:
                return NodeType.Section
            else:
                return NodeType.Document

        return NodeType.Metadata    


    def get_graph(self, nodes, edges):
        NODE = 'node'

        # Build DGL graph
        graph_data = {}

        # edge_types = {}

        # Process edges
        edgetype2tensor1, edgetype2tensor2, edge_types = {}, {}, set()
        for n1, edge_type, n2 in edges:
            node1_index = nodes.index(n1)
            node2_index = nodes.index(n2)
            if not edge_type in edgetype2tensor1: edgetype2tensor1[edge_type] = []
            if not edge_type in edgetype2tensor2: edgetype2tensor2[edge_type] = []
            edgetype2tensor1[edge_type].append(node1_index)
            edgetype2tensor2[edge_type].append(node2_index)
            edge_types.add(edge_type)

            # node1_type = self.get_node_type(n1)
            # node2_type = self.get_node_type(n2)
            # if (node1_type, edge_type, node2_type) in graph_data:
            #     graph_embeddings[(node1_type, edge_type, node2_type)]

        for edge_type in edge_types:
            graph_data[(NODE, edge_type, NODE)] = (torch.tensor(edgetype2tensor1[edge_type]),
                                                torch.tensor(edgetype2tensor2[edge_type]))


        # Finalize the graph
        G = dgl.heterograph(graph_data)
        # print(G)

        G = dgl.to_homogeneous(G)
        G = dgl.add_self_loop(G)

        # print(G)

        #print(G.canonical_etypes)
        assert(G.number_of_nodes() == len(nodes))
        #print(G)
        #G = dgl.to_bidirected(G)

        #print(G)
        return G


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_sentence_embeddings(self, sents, device, is_project=False):
        sents = [s for s in sents if s]
        encoded_input = self.scibert_tokenizer(sents, padding='max_length', truncation=True, return_tensors='pt', max_length=100).to(device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.scibert(**encoded_input)

        # Perform pooling. In this case, mean pooling
        pool = self.mean_pooling(model_output, encoded_input['attention_mask'])

        if is_project:
            m = nn.Linear(768, 50).to(device)
            return(m(pool))
        else:
            return pool

    def forward(self, idx, split, device):
        
        graph_info = self.graphs[split][idx]
        nodes = graph_info['nodes']
        edges = graph_info['edges']
        article_id = graph_info['id']

        device = f"cuda:{device}"
        self.scibert = self.scibert.to(device)
        self.GM = self.GM.to(device)
        
        # get initial embeddings
        final_nodes, embeddings = self.get_initial_embeddings(article_id, nodes, edges, device)
        final_edges = [e for e in edges if (e[0] in final_nodes and e[2] in final_nodes)]
        
        embeddings = torch.stack(embeddings).to(device)
        G = self.get_graph(final_nodes, final_edges).to(device)

        graph_embeddings = self.GM(G, embeddings)
        
        return graph_embeddings
    

@register_model("bart")
class BARTModel(TransformerModel):
    __jit_unused_properties__ = ["supported_targets"]

    @classmethod
    def hub_models(cls):
        return {
            "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
            "bart.large": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz",
            "bart.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz",
            "bart.large.cnn": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz",
            "bart.large.xsum": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz",
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()
        
        if args.dual_graph_encoder:
            self.graph_encoder = GraphEncoder()
            self.graph_cross_attention = torch.nn.MultiheadAttention(1024, 4)


        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()


    @staticmethod
    def add_args(parser):
        super(BARTModel, BARTModel).add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )

        parser.add_argument(
            '--restrict-position-embed',
            action='store_true',
            default=False,
            help="do no extend the position embeddings"
        )

        parser.add_argument(
            '--sliding-window', 
            action='store_true', 
            default=False, 
            help="use sliding window attention as in longformer",
        )
        parser.add_argument(
            '--block-attention', 
            action='store_true', 
            default=False, 
            help="use block attention",
        )
        parser.add_argument(
            '--top-down', 
            action='store_true', 
            default=False, 
            help="use block attention",
        )

        parser.add_argument(
            '--dual_graph_encoder', 
            action='store_true', 
            default=False, 
            help="use dual graph encoder",
        )
        
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        # use vanilla attention for now
        args.use_xformers = False

        return super().build_decoder(
            TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        # if args.sliding_window:
        #     from fairseq.models.long_transformers.sliding_window import SWTransformerEncoder
        #     args.attention_window = [512]
        #     return SWTransformerEncoder(args, src_dict, embed_tokens)
        if args.block_attention:
            args.window_size = 1024
            from fairseq.models.long_transformers.block import BlockTransformerEncoder
            return BlockTransformerEncoder(args, src_dict, embed_tokens)
        # if args.top_down:
        #     args.window_size = 1024
        #     args.encoder_n1 = 8
        #     args.encoder_n2 = 2
        #     args.encoder_n3 = 4
        #     from fairseq.models.long_transformers.top_down import TopDownTransformerEncoder
        #     return TopDownTransformerEncoder(args, src_dict, embed_tokens)    
        return super().build_encoder(
            TransformerConfig.from_namespace(args), src_dict, embed_tokens
        )
        
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg = None,
        args = None,
    ):  
        # if self.args.top_down:
        #     strict=False
        return super().load_state_dict(state_dict, strict, model_cfg, args)

    @property
    def supported_targets(self):
        return {"self"}

    def forward(
        self,
        split,
        aids,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        masked_unfiltered: Optional[torch.Tensor] = None,
        # src_graph_edges: Optional[torch.Tensor] = None,
        # src_graph_nodes: Optional[torch.Tensor] = None,
    ):
        if classification_head_name is not None:
            features_only = True

        # print("IDs: ", aids)

        # print("ARGS: ", self.args)

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings, # if token representation updating, somehow use this
            return_all_hiddens=return_all_hiddens
        )

        # If dual encoding, encode the article graph and then update encoder_out
        if self.args.dual_graph_encoder:
            enc_output = encoder_out['encoder_out']
            device = aids.get_device()
            self.graph_cross_attention = self.graph_cross_attention.to(device)

            graph_enc_out = []
            for i, aid in enumerate(aids):
                graph_out = self.graph_encoder.forward(aid, split, device)
                # print(graph_out.shape)
                # pad graph output to 1024
                graph_out = torch.nn.functional.pad(graph_out, (0,0,0,1024-graph_out.shape[0]), "constant", 0)
                graph_enc_out.append(graph_out)
            

            # print("encoder", enc_output[0].shape)
            graph_enc_out = torch.stack(graph_enc_out, dim=1).to(torch.float16)

            # print("graph ", graph_enc_out.shape)

            attn_output, attn_output_weights = self.graph_cross_attention(enc_output[0], graph_enc_out, graph_enc_out)


            # print("Attn output: ", attn_output.shape)

            encoder_out['encoder_out'] = [attn_output]
            
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        eos: int = self.eos
        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(eos), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    x = head(sentence_representation)
                    break
        return x, extra

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        sample_break_mode="eos",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            sample_break_mode=sample_break_mode,
            **kwargs,
        )
        return BARTHubInterface(x["args"], x["task"], x["models"][0])

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            do_spectral_norm=getattr(
                self.args, "spectral_norm_classification_head", False
            ),
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
            loaded_dict_size == len(self.encoder.dictionary) + 1
            and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        # When continued pretraining on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
            self.encoder.dictionary
        ):
            logger.info(
                "Adding extra language embeddings not found in pretrained model for "
                "continued pretraining of MBART on new set of languages."
            )
            loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
                -1, :
            ]

            num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["encoder.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["encoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )
            state_dict["decoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["decoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def safe_getattr(obj, k, default=None):
    from omegaconf import OmegaConf

    if OmegaConf.is_config(obj):
        return obj[k] if k in obj and obj[k] is not None else default

    return getattr(obj, k, default)

@register_model_architecture("bart", "bart_large")
def bart_large_architecture(args):

    # # @xwhan has to put it here due to a bug
    # def getattr(args, key, value):
    #     return value

    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)

    args.max_target_positions = safe_getattr(args, "max_target_positions", 1024) #hack
    args.max_source_positions = safe_getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)


@register_model_architecture("bart", "bart_prelayernorm")
def bart_prelayernorm_architecture(args):

    def getattr(args, key, value):
        return value

    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = safe_getattr(args, "max_target_positions", 1024)
    args.max_source_positions = safe_getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)


@register_model_architecture("bart", "bart_base")
def bart_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)

@register_model_architecture("bart", "bart_xlarge")
def bart_base_architecture(args):
    bart_large_architecture(args)
    args.encoder_layers = 24
    args.decoder_layers = 24

@register_model_architecture("bart", "bart_slarge")
def bart_base_architecture(args):
    bart_large_architecture(args)
    args.encoder_layers = 16
    args.decoder_layers = 16

@register_model_architecture("bart", "mbart_large")
def mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)


@register_model_architecture("bart", "mbart_base")
def mbart_base_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_base_architecture(args)


@register_model_architecture("bart", "mbart_base_wmt20")
def mbart_base_wmt20_architecture(args):
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    mbart_base_architecture(args)
