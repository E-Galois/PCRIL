import scipy.misc
import scipy.io
from ops import *
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
from time import time
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

clip_model, img_preprocess = clip.load("ViT-B/32", device='cuda')
clip_model.eval()
for name, param in clip_model.named_parameters():
    param.requires_grad_(False)
print('turned off CLIP gradient')
_tokenizer = _Tokenizer()


def init_parameters_recursively(layer):
    if isinstance(layer, nn.Sequential):
        for sub_layer in layer:
            init_parameters_recursively(sub_layer)
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, std=0.01)
        if layer.bias is not None:
            nn.init.normal_(layer.bias, std=0.01)
    else:
        return


class LabelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding

    @property
    def dtype(self):
        return clip_model.visual.conv1.weight.dtype

    def forward(self, prompts, eots):
        x = prompts.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # ND * Dd = Nd
        x = x[torch.arange(x.shape[0]), eots] @ self.text_projection
        return x


label_encoder = LabelEncoder()


class LabelNet(nn.Module):
    def __init__(self, cfg):
        super(LabelNet, self).__init__()
        self.bit = cfg.bit
        self.n_cls = cfg.numClass
        self.cls_vec = cfg.cls_vec

        self.n_pre_agno = 2
        self.n_prefix = 2
        self.n_suffix = 2
        self.n_suf_agno = 2

        self.pre_agno = nn.Parameter(torch.randn(self.n_pre_agno, clip_model.token_embedding.embedding_dim, dtype=torch.float32))
        self.suf_agno = nn.Parameter(torch.randn(self.n_suf_agno, clip_model.token_embedding.embedding_dim, dtype=torch.float32))
        self.prefix = nn.Parameter(torch.randn(self.n_cls, self.n_prefix, clip_model.token_embedding.embedding_dim, dtype=torch.float32))
        self.suffix = nn.Parameter(torch.randn(self.n_cls, self.n_suffix, clip_model.token_embedding.embedding_dim, dtype=torch.float32))
        prefix_fill = ' '.join(['X'] * self.n_prefix)
        suffix_fill = ' '.join(['X'] * self.n_suffix)
        self.cls_repr_lens = np.array([len(_tokenizer.encode(name)) for name in self.cls_vec])
        cls_embed = [prefix_fill + " " + name + " " + suffix_fill for name in self.cls_vec]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in cls_embed]).cuda()  # X X class name X X
        with torch.no_grad():
            self.embeddings = clip_model.token_embedding(tokenized_prompts).float()  # (n_cls, 77, 256)
        #for cls_i in range(self.n_cls):
        #    self.embeddings[cls_i, 1: 1 + self.n_prefix, :] = self.prefix[cls_i, :, :]
        #    self.embeddings[cls_i, 1 + self.n_prefix + self.cls_repr_lens[cls_i]: 1 + self.n_prefix + self.cls_repr_lens[cls_i] + self.n_suffix, :] = self.suffix[cls_i, :, :]
        blank = clip.tokenize('').cuda()  # SOS EOS ...
        self.n_ctx = blank.shape[1]
        with torch.no_grad():
            self.register_buffer('blank_prompt', clip_model.token_embedding(blank).float()[0])  # (77, 256)
        # self.label_encoder = LabelEncoder()
        # self.init_parameters()

    def init_parameters(self):
        # init_parameters_recursively(self.hash)
        pass

    def cls_ind_to_prompt(self, ind):
        prompt_frags = []
        prompt_frags.append(self.blank_prompt[0:1, :])
        n_len = 1
        prompt_frags.append(self.pre_agno)
        n_len += self.n_pre_agno
        try:
            for i in ind:
                cls_frag = self.embeddings[i, 1: 1 + self.n_prefix + self.cls_repr_lens[i] + self.n_suffix, :].detach()
                cls_frag[:self.n_prefix, :] = self.prefix[i, :, :]
                cls_frag[self.n_prefix + self.cls_repr_lens[i]:, :] = self.suffix[i, :, :]
                prompt_frags.append(cls_frag)
                n_len += cls_frag.shape[0]
        except:
            print("?")
        prompt_frags.append(self.suf_agno)
        n_len += self.n_suf_agno
        n_rem = self.n_ctx - n_len
        prompt_frags.append(self.blank_prompt[1:1 + n_rem, :])
        return torch.cat(prompt_frags, dim=0), n_len

    def encode(self, ind_list):
        prompts = []
        eots = []
        for ind in ind_list:
            prompt, eot = self.cls_ind_to_prompt(ind)
            prompts.append(prompt)
            eots.append(eot)
        prompts = torch.stack(prompts)  # n, 77, 256
        clip_l = label_encoder(prompts, eots).float()
        return clip_l

    def forward(self, label_ic):
        ind_all = []
        pos_all = []
        neg_all = []
        pos_eots = []
        neg_eots = []
        for i in range(label_ic.shape[0]):
            l_ic = label_ic[i]
            pos_i = torch.where(l_ic == 1.0)[0]
            neg_i = torch.where(l_ic == 0.0)[0]
            np_i = len(pos_i)
            nn_i = len(neg_i)
            if np_i == 0:
                continue
            #strong_i = pos_i[torch.randperm(np_i)[0:torch.randint(low=1, high=np_i + 1, size=(1,))]]
            strong_i = pos_i[torch.randperm(np_i)[0:torch.randint(low=1, high=np_i + 1, size=(1,))].sort()[0]]
            strong_prompt, pos_eot = self.cls_ind_to_prompt(strong_i)
            # =========== (strong, weak) ===========
            weak_i = strong_i[:-1]
            weak_prompt, eot = self.cls_ind_to_prompt(weak_i)
            ind_all.append(i)
            neg_all.append(weak_prompt)
            neg_eots.append(eot)
            pos_all.append(strong_prompt)
            pos_eots.append(pos_eot)
            if nn_i == 0:
                continue
            # =========== (strong, wrong) ===========
            #strong_i_tmp = pos_i[torch.randperm(np_i)]
            weak_i_tmp = strong_i[:-1]
            wrong_i_j = torch.randint(low=0, high=nn_i, size=(1,))
            wrong_i = torch.cat([weak_i_tmp, neg_i[wrong_i_j: wrong_i_j + 1]], dim=-1)
            wrong_prompt, eot = self.cls_ind_to_prompt(wrong_i)
            ind_all.append(i)
            neg_all.append(wrong_prompt)
            neg_eots.append(eot)
            pos_all.append(strong_prompt)
            pos_eots.append(pos_eot)
            # =========== (strong, exceed) ===========
            exceed_i_j = torch.randint(low=0, high=nn_i, size=(1,))
            exceed_i = torch.cat([strong_i, neg_i[exceed_i_j: exceed_i_j + 1]], dim=-1)
            exceed_prompt, eot = self.cls_ind_to_prompt(exceed_i)
            ind_all.append(i)
            neg_all.append(exceed_prompt)
            neg_eots.append(eot)
            pos_all.append(strong_prompt)
            pos_eots.append(pos_eot)
        n_all = len(pos_all)
        if n_all != 0:
            pos_all = torch.stack(pos_all)  # n, 77, 256
            neg_all = torch.stack(neg_all)  # n, 77, 256
            clip_pos = label_encoder(pos_all, pos_eots).float()
            clip_neg = label_encoder(neg_all, neg_eots).float()
            return ind_all, clip_pos, clip_neg  # not normalized
        else:
            return None, None, None


class ImageNetMIDef(nn.Module):
    def __init__(self, cfg):
        super(ImageNetMIDef, self).__init__()
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.bit = cfg.bit
        self.k = 5
        self.numClass = cfg.numClass
        self.feature = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.SEMANTIC_EMBED),
            nn.ReLU(),
        )
        self.hash = nn.Sequential(
            nn.Linear(self.SEMANTIC_EMBED, self.bit),
            nn.Tanh()
        )
        '''self.label = nn.Sequential(
            nn.Linear(self.SEMANTIC_EMBED, self.numClass),
            nn.Sigmoid()
        )'''
        self.blend_coef = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.feature)
        init_parameters_recursively(self.hash)
        #init_parameters_recursively(self.label)

    def get_coef(self):
        return self.blend_coef.detach()

    def forward(self, clip_i, clip_i_mix=None):
        with torch.no_grad():
            clip_i = clip_model.encode_image(clip_i).float().view(clip_i.shape[0], -1)
            if clip_i_mix is not None:
                clip_i_mix = clip_model.encode_image(clip_i_mix).float().view(clip_i.shape[0], -1)
                clip_i_mix = self.blend_coef * clip_i + (1 - self.blend_coef) * clip_i_mix
                clip_i = torch.cat([clip_i, clip_i_mix], dim=0)
        fea_I = self.feature(clip_i)
        hsh_I = self.hash(fea_I)
        #lab_I = self.label(fea_I)
        return torch.squeeze(clip_i), torch.squeeze(fea_I), torch.squeeze(hsh_I)#, torch.squeeze(lab_I)

    def get_hash(self, feature):
        hsh_I = self.hash(feature)
        return torch.squeeze(hsh_I)

    def get_clip_feature(self, inputs):
        base = clip_model.encode_image(inputs).float().view(inputs.shape[0], -1)
        return base / base.norm(dim=-1, keepdim=True)


class TextNetMIDef(nn.Module):
    def __init__(self, cfg):
        super(TextNetMIDef, self).__init__()
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.bit = cfg.bit
        self.k = 5
        self.numClass = cfg.numClass
        # self.dimTxt = cfg.dimTxt
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=2048, out_channels=self.SEMANTIC_EMBED, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
        )
        self.norm = nn.modules.normalization.LocalResponseNorm(size=4, alpha=0.0001, beta=0.75, k=2.0)
        self.hash = nn.Sequential(
            nn.Conv2d(in_channels=self.SEMANTIC_EMBED, out_channels=self.bit, kernel_size=(1, 1), stride=(1, 1)),
            nn.Tanh(),
        )
        '''self.label = nn.Sequential(
            nn.Conv2d(in_channels=self.SEMANTIC_EMBED, out_channels=self.numClass, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )'''
        self.blend_coef = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.feature)
        init_parameters_recursively(self.hash)
        # init_parameters_recursively(self.label)

    def get_coef(self):
        return self.blend_coef.detach()

    def forward(self, clip_t, clip_t_mix=None):
        with torch.no_grad():
            clip_t = clip_model.encode_text(clip_t).float()
            clip_t = clip_t.unsqueeze(-1).unsqueeze(-1)
            if clip_t_mix is not None:
                clip_t_mix = clip_model.encode_text(clip_t_mix).float()
                clip_t_mix = clip_t_mix.unsqueeze(-1).unsqueeze(-1)
                clip_t_mix = self.blend_coef * clip_t + (1 - self.blend_coef) * clip_t_mix
                clip_t = torch.cat([clip_t, clip_t_mix], dim=0)
        fea_T = self.feature(clip_t)
        norm = self.norm(fea_T)
        hsh_T = self.hash(norm)
        #lab_T = self.label(norm)
        tuple = torch.squeeze(clip_t), torch.squeeze(fea_T), torch.squeeze(hsh_T)#, torch.squeeze(lab_T)
        return tuple

    def get_hash(self, feature):
        fea_T = feature.reshape([feature.shape[0], feature.shape[1], 1, 1])
        norm = self.norm(fea_T)
        hsh_T = self.hash(norm)
        return torch.squeeze(hsh_T)

    def get_clip_feature(self, inputs):
        inputs = clip_model.encode_text(inputs).float()
        return inputs / inputs.norm(dim=-1, keepdim=True)


class FMUN(nn.Module):
    def __init__(self, cfg):
        super(FMUN, self).__init__()
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.bit = cfg.bit
        self.inet = ImageNetMIDef(cfg)
        self.tnet = TextNetMIDef(cfg)

    def forward(self, image, text, image2, text2):
        bs = image.shape[0]
        bs_ = text.shape[0]
        assert bs == bs_
        clip_i, fea_i, hsh_i = self.inet(image, image2)
        clip_t, fea_t, hsh_t = self.tnet(text, text2)
        #clip_i, fea_i, hsh_i = self.inet(image)
        #clip_t, fea_t, hsh_t = self.tnet(text)
        return hsh_i[:bs], hsh_t[:bs], hsh_i[bs:], hsh_t[bs:]
        #return hsh_i, hsh_t

    def get_image_hash(self, image):
        clip_i, fea_i, hsh_i = self.inet(image)
        return hsh_i

    def get_text_hash(self, text):
        clip_t, fea_t, hsh_t = self.tnet(text)
        return hsh_t
