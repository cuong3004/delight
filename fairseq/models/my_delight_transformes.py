# import math
#
# from fairseq.delight_modules.dextra_emb import DExTraEmb
# from fairseq.delight_modules.nn_functions import get_embedding_layer
# from fairseq.models.delight_transformer import DeLighTTransformerModel
# from fairseq.distributed_utils import is_master
# from fairseq.delight_modules import (
#     DEFAULT_WIDTH_MULTIPLIER,
#     DEFAULT_MIN_DEXTRA_LAYERS,
#     MIN_ELEMENTS_PER_GROUP,
#     DEFAULT_FFN_RED_FACTOR,
#     DEFAULT_DROPOUT,
#     DEFAULT_MAX_DEXTRA_LAYERS
# )
#
# DEFAULT_MAX_SOURCE_POSITIONS = 1024
# DEFAULT_MAX_TARGET_POSITIONS = 1024
#
# class MyDeLighTTransformerModel(DeLighTTransformerModel):
#     @classmethod
#     def build_my_model(cls, args, src_length, src_pad_idx, tgt_length, tgt_pad_idx):
#         """Build a new model instance."""
#
#         # make sure all arguments are present in older models
#         base_architecture(args)
#
#         if getattr(args, "max_source_positions", None) is None:
#             args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
#         if getattr(args, "max_target_positions", None) is None:
#             args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
#
#         # src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
#
#         def build_embedding(args, vocab_length, vocab_pad_idx):
#             num_embeddings = vocab_length
#             padding_idx = vocab_pad_idx
#             if args.adaptive_input:
#                 raise ValueError('Adaptive Input is not yet supported for NMT')
#             else:
#                 map_layer = get_embedding_layer(num_embeddings=num_embeddings,
#                                                 embedding_dim=args.delight_emb_map_dim,
#                                                 padding_idx=padding_idx)
#
#             emb = DExTraEmb(args, map_layer=map_layer)
#
#             return emb
#
#         encoder_embed_tokens = build_embedding(args, src_length, src_pad_idx)
#         decoder_embed_tokens = build_embedding(args, tgt_length, tgt_pad_idx)
#         if args.share_all_embeddings:
#             if args.adaptive_input:
#                 raise ValueError('Adaptive Input is not yet supported for NMT')
#             else:
#                 decoder_embed_tokens.map_layer.weight = encoder_embed_tokens.map_layer.weight
#
#         encoder = cls.build_encoder(args, None, encoder_embed_tokens)
#         decoder = cls.build_decoder(args, [0]*tgt_length, decoder_embed_tokens)
#
#         # print macs and params layer-wise
#         if args.print_stats and is_master(args):
#             cls.comptue_stats(args, encoder, decoder)
#
#         return cls(args, encoder, decoder)
#
# def base_architecture(args):
#     # DeLighT Embedding layer
#     args.adaptive_input = getattr(args, "adaptive_input", False)
#     args.delight_emb_map_dim = getattr(args, "delight_emb_map_dim", 128)
#     args.delight_emb_out_dim = getattr(args, "delight_emb_out_dim", 128)
#     # compute the max groups in GLT
#     assert args.delight_emb_out_dim % MIN_ELEMENTS_PER_GROUP == 0, 'remainder({}, {}) should be equal to 0'.format(
#         args.delight_emb_out_dim, MIN_ELEMENTS_PER_GROUP)
#     max_groups = 2 ** math.ceil(math.log(args.delight_emb_out_dim // MIN_ELEMENTS_PER_GROUP, 2))
#
#     args.delight_emb_max_groups = getattr(args, "delight_emb_max_groups", max_groups)
#     args.delight_emb_dropout = getattr(args, "delight_emb_dropout", DEFAULT_DROPOUT)
#     args.delight_emb_depth = getattr(args, "delight_emb_depth", DEFAULT_MIN_DEXTRA_LAYERS)
#     args.delight_emb_width_mult = getattr(args, "delight_emb_width_mult", DEFAULT_WIDTH_MULTIPLIER)
#
#     # Encoder arguments in DeLighT
#     args.delight_enc_scaling = getattr(args, "delight_enc_scaling", 'block')
#     args.delight_enc_layers = getattr(args, "delight_enc_layers", DEFAULT_MAX_DEXTRA_LAYERS)
#     args.delight_enc_min_depth = getattr(args, "delight_enc_min_depth", DEFAULT_MIN_DEXTRA_LAYERS)
#     args.delight_enc_max_depth = getattr(args, "delight_enc_max_depth", DEFAULT_MAX_DEXTRA_LAYERS)
#     args.delight_enc_width_mult = getattr(args, "delight_enc_width_mult", DEFAULT_WIDTH_MULTIPLIER)
#     args.delight_enc_ffn_red = getattr(args, "delight_enc_ffn_red", DEFAULT_FFN_RED_FACTOR)
#     args.delight_enc_max_groups = getattr(args, "delight_enc_max_groups", max_groups)
#
#     # Decoder arguments in DeLighT
#     args.delight_dec_scaling = getattr(args, "delight_dec_scaling", 'block')
#     args.delight_dec_layers = getattr(args, "delight_dec_layers", DEFAULT_MAX_DEXTRA_LAYERS)
#     args.delight_dec_min_depth = getattr(args, "delight_dec_min_depth", DEFAULT_MIN_DEXTRA_LAYERS)
#     args.delight_dec_max_depth = getattr(args, "delight_dec_max_depth", DEFAULT_MAX_DEXTRA_LAYERS)
#     args.delight_dec_width_mult = getattr(args, "delight_dec_width_mult", DEFAULT_WIDTH_MULTIPLIER)
#     args.delight_dec_ffn_red = getattr(args, "delight_dec_ffn_red", DEFAULT_FFN_RED_FACTOR)
#     args.delight_dec_max_groups = getattr(args, "delight_dec_max_groups", max_groups)
#
#     ## Others
#     args.no_glt_shuffle = getattr(args, "no_glt_shuffle", False)
#     args.glt_shuffle = not args.no_glt_shuffle
#     args.define_iclr = getattr(args, "define_iclr", False)
#     args.delight_dropout = getattr(args, "delight_dropout", DEFAULT_DROPOUT)
#
#     # normalization and activation layers
#     args.norm_type = getattr(args, "norm_type", 'ln')
#     args.act_type = getattr(args, "act_type", 'swish')
#
#     # ADAPTIVE INPUT AND OUTPUT PARAMS
#     args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
#     args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
#     args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
#     args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
#     args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)
#
#     # Print  stats
#     args.print_stats = getattr(args, "print_stats", False)
#     args.src_len_ps = getattr(args, "src_len_ps", 20)
#     args.tgt_len_ps = getattr(args, "tgt_len_ps", 20)
#
#     # DROPOUTS
#     args.attention_dropout = getattr(args, "attention_dropout", DEFAULT_DROPOUT)
#     args.activation_dropout = getattr(args, "activation_dropout", 0.0)
#     args.dropout = getattr(args, "dropout", DEFAULT_DROPOUT)
#     args.delight_dropout = getattr(args, "delight_dropout", 0.0)
#     args.pe_dropout = getattr(args, "pe_dropout", DEFAULT_DROPOUT)
#     args.ffn_dropout = getattr(args, "ffn_dropout", DEFAULT_DROPOUT)
#
#     # Other parameters
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
#     args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
#
#     args.share_decoder_input_output_embed = getattr(
#         args, "share_decoder_input_output_embed", False
#     )
#     args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
#     args.no_token_positional_embeddings = getattr(
#         args, "no_token_positional_embeddings", False
#     )
#     args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
#     args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
#
#     args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
