import torch as th
import torch.nn as nn
import torch.nn.functional as F

from nn import timestep_embedding
from UNet2 import UNetModel
from xf import LayerNorm, Transformer, convert_module_to_f16
from transformers import EsmTokenizer, EsmModel, EsmConfig

class Text2ImUNet(UNetModel):
    """
    A UNetModel that conditions on text with an encoding transformer.

    Expects an extra kwarg `tokens` of text.

    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    :param tokenizer: the text tokenizer for sampling/vocab size.
    """

    def __init__(
        self,
        text_ctx,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        tokenizer,
        *args,
        cache_text_emb=False,
        xf_ar=0.0,
        xf_padding=False,
        share_unemb=False,
        **kwargs,
    ):
        # self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_ar = xf_ar
        self.xf_padding = xf_padding
        self.tokenizer = tokenizer

        if not xf_width:
            super().__init__(*args, **kwargs, encoder_channels=None)
        else:
            super().__init__(*args, **kwargs, encoder_channels=xf_width)
        # if self.xf_width:
        #     self.transformer = Transformer(
        #         text_ctx,
        #         xf_width,
        #         xf_layers,
        #         xf_heads,
        #     )
        #     if xf_final_ln:
        #         self.final_ln = LayerNorm(xf_width)
        #     else:
        #         self.final_ln = None

        #     self.token_embedding = nn.Embedding(self.tokenizer.n_vocab, xf_width)
        #     self.positional_embedding = nn.Parameter(th.empty(text_ctx, xf_width, dtype=th.float32))
        self.transformer_proj = nn.Linear(xf_width, self.model_channels * 4)

        #     if self.xf_padding:
        #         self.padding_embedding = nn.Parameter(
        #             th.empty(text_ctx, xf_width, dtype=th.float32)
        #         )
        #     if self.xf_ar:
        #         self.unemb = nn.Linear(xf_width, self.tokenizer.n_vocab)
        #         if share_unemb:
        #             self.unemb.weight = self.token_embedding.weight
        # self.down_proj_prot = nn.Linear(self.xf_width, self.model_channels)

        # Load ESM configuration for the specific model version you want
        config = EsmConfig.from_pretrained("facebook/esm2_t6_8M_UR50D")
        config.num_attention_heads = 5
        config.hidden_size = 160
        config.intermediate_size = 640
        config.num_hidden_layers = 2
        # print(config)
        # exit()


        # self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        # Create the model using the architecture but with random weights (no pre-trained weights)
        self.esm_model = EsmModel(config).to('cuda')

        self.cache_text_emb = cache_text_emb
        self.cache = None

    def convert_to_fp16(self):
        super().convert_to_fp16()
        if self.xf_width:
            self.transformer.apply(convert_module_to_f16)
            self.transformer_proj.to(th.float16)
            self.token_embedding.to(th.float16)
            self.positional_embedding.to(th.float16)
            if self.xf_padding:
                self.padding_embedding.to(th.float16)
            if self.xf_ar:
                self.unemb.to(th.float16)

    def get_text_emb(self, ids, attn):        
        # if protein_string[0] == "":
        #     xf_out = th.zeros([len(protein_string), 1024, 160], device="cuda")
        #     xf_proj = self.transformer_proj(xf_out[:, -1])
        #     xf_out = xf_out.permute(0, 2, 1)
        #     outputs = dict(xf_proj=xf_proj, xf_out=xf_out)
        #     return outputs
        
        # print(protein_string)
        # print(th.cuda.mem_get_info())
        # print(ids.shape)
        # print(attn.shape)
        xf_out = self.esm_model(input_ids=ids, attention_mask=attn).last_hidden_state
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL
        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)
        return outputs


    def del_cache(self):
        self.cache = None

        

    def forward(self, x, timesteps, ids, attn):
        hs = []
        # print(x.shape)
        # exit()
        # xf_proj = self.down_proj_prot(xf_proj)
        # print(xf_proj.shape)        
        # xf_proj = xf_proj.squeeze(1)

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # print(emb.shape)
        # print(xf_proj.shape)
        # xf_proj = xf_proj.squeeze(1)
        # th.cuda.empty_cache()

        # a = th.cuda.memory_allocated(0) / 10e9
        # r = th.cuda.memory_reserved(0) / 10e9
        # print(f"Mem before {(r - a) / 10e9}")
        # print(th.cuda.mem_get_info())

        protein_outputs = self.get_text_emb(ids, attn)

        # a = th.cuda.memory_allocated(0) / 10e9
        # r = th.cuda.memory_reserved(0) / 10e9
        # print(f"Mem after {(r - a) / 10e9}")
        # print(th.cuda.mem_get_info())

        xf_proj, xf_out = protein_outputs["xf_proj"], protein_outputs["xf_out"]

        emb = emb + xf_proj.to(emb)

        h = x.type(self.dtype)
        # print(emb.shape)
        # exit()
        # print("EMB", emb.shape)

        # exit()

        # print(h.shape)
        # print(xf_proj.shape)
        # exit()
        for module in self.input_blocks:
            # print(module)
            # print(xf_proj.shape)
            # print(emb.shape)
            # exit()
            h = module(h, emb, xf_out)
            hs.append(h)
            # print(h.shape)
        
        h = self.middle_block(h, emb, xf_out)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        return h