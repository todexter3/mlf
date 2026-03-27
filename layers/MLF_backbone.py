__all__ = ['MLF_backbone']

# Cell
import time
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from layers.MLF_layers import *
import numpy as np
import torch.nn.functional as f
from layers.RevIN import RevIN

class ForecastingBlock(nn.Module):
    def __init__(self, thetas_dim, backcast_length=10, forecast_length=5, configs=None):
        super().__init__()
        self.configs = configs
        self.thetas_dim=thetas_dim
        hid_dim=thetas_dim
        units=thetas_dim//2
        self.fc1 = nn.Linear(thetas_dim, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.theta_b_fc = nn.Linear(units, hid_dim, bias=False)
        self.theta_f_fc = nn.Linear(units, hid_dim, bias=False)

        self.backcast_fc = nn.Linear(units, backcast_length)
        self.forecast_fc = nn.Linear(units, forecast_length)

        self.flatten = nn.Flatten(start_dim=-2)
        self.actv={'relu':F.relu,'sft':F.softplus}
    def forward(self, x_or):
        act = 'relu'
        if self.configs.activation_tag:
            x=self.actv[act](self.fc1(x_or))
        else:
            x = self.fc1(x_or)
        backcast = self.backcast_fc(x)
        forecast = self.forecast_fc(x)
        return backcast,forecast

class MLF_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False,configs=None, **kwargs):
        super().__init__()
        self.configs=configs
        self.padding_patch = padding_patch
        patch_num_all_scale=[]
        patch_len_all_scale=[]
        self.padding_patch_layer_all_scale=[]

        for i,p_s in enumerate(self.configs.patchLen_stride_all):
            patch_len_all_scale.append(p_s[0])
            if padding_patch == 'end':
                patch_num_all_scale.append(int((self.configs.scal_all[i] - p_s[0])/p_s[1] + 1)+1)
                self.padding_patch_layer_all_scale.append(nn.ReplicationPad1d((0, p_s[1])) )
            else:
                patch_num_all_scale.append(int((self.configs.scal_all[i] - p_s[0]) / p_s[1] + 1))
        self.configs.patch_len_all_scale=patch_len_all_scale

        self.target_window=target_window
        self.flatten = nn.Flatten(start_dim=-2)
        self.scale_num = len(self.configs.scal_all)
        self.configs.p_len_embed_size=d_model

        patch_num_all=sum(patch_num_all_scale)

        self.configs.patch_num_all_scale=patch_num_all_scale
        # Backbone
        self.configs.head_nf_res_all=[]

        self.configs.target_window=target_window
        self.configs.scale_num=self.scale_num
        self.head_nf = d_model * patch_num_all
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        # if self.configs.mode=='long-term':
        #     self.configs.threshold_patch_num=2
        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            if self.configs.patch_squeeze:
                patch_num_all_scale_squeeze=[]
                for i,p_n in enumerate(patch_num_all_scale) :
                    if p_n<=self.configs.threshold_patch_num:
                        patch_num_all_scale_squeeze.append(p_n//1)
                    else:
                        patch_num_all_scale_squeeze.append(p_n // self.configs.squeeze_factor[i])
                self.configs.patch_num_all_scale_squeeze=patch_num_all_scale_squeeze
                self.head_nf = d_model * sum(patch_num_all_scale_squeeze)

                self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout,args=self.configs)
            else:
                self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout, args=self.configs)
        self.flat=nn.Flatten(1)

        self.actv={'relu':F.relu,'sft':F.softplus}

        if self.configs.patch_squeeze:
            for i in range(self.scale_num):
                self.configs.head_nf_res_all.append(d_model*self.configs.patch_num_all_scale_squeeze[i])
        else:

            for i in range(self.scale_num):
                self.configs.head_nf_res_all.append(d_model*self.configs.patch_num_all_scale[i])

        self.backbone = TSTiEncoder(c_in,configs_=self.configs, patch_num=patch_num_all, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)


    def forward(self, z_all_or):                                                                   # z: [bs x nvars x seq_len]
        norm_all={}
        b_s,n_var,seq_len=z_all_or[0].shape
        self.configs.n_var=n_var
        self.configs.b_s = b_s
        for i, z in enumerate(z_all_or):
            norm_all[i] = z
        if self.padding_patch == 'end':
            for k,v in norm_all.items():

                norm_all[k]=self.padding_patch_layer_all_scale[k](v)
        for k, v in norm_all.items():
            norm_all[k] = v.unfold(dimension=-1, size=self.configs.patchLen_stride_all[k][0], step=self.configs.patchLen_stride_all[k][1])
            if self.configs.patch_pad:
                if norm_all[k].shape[-1]!=self.configs.max_patch_len and not self.configs.MAP :
                    pad_size1 = self.configs.max_patch_len - norm_all[k].size(-1)
                    norm_all[k] = torch.nn.functional.pad(norm_all[k], (0, pad_size1)).permute(0,1,3,2)
                else:
                    norm_all[k]=norm_all[k].permute(0,1,3,2)
        #z_all_or is used for LWI module
        z,scale_all_rec,scale_all_patch = self.backbone(norm_all,x_all_or=z_all_or)
        if self.configs.revin_norm:
            z = z.permute(0, 2, 1)
            z = self.configs.revin_layer(z, 'denorm')
        else:
            z = z.permute(0, 2, 1)
        return z,scale_all_rec,scale_all_patch



class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0,args=None):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.args=args
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)

            # self.fc1 = nn.Linear(nf, nf//2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x) #torch.Size([32, 7, 6144])
            x = self.linear(x)
            x = self.dropout(x)

        return x

class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, configs_=None,verbose=False, **kwargs):
        super().__init__()
        self.configs=configs_
        self.patch_num = patch_num
        # Input encoding
        q_len = patch_num


        self.seq_len = q_len
        self.configs.q_len=q_len
        params_dict = vars(self.configs)
        for key, value in params_dict.items():
            if isinstance(value, np.ndarray):
                params_dict[key] = value.tolist()
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.scale_num = self.configs.scale_num
        # Encoder
        self.encoder = MLFEncoder(q_len, d_model, n_heads,configs_=configs_, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        self.flatten = nn.Flatten(start_dim=-2)

        self.configs.count = 0
        self.individually_embed=True
        self.reduce_fac=self.configs.squeeze_factor

        ##PatschSqueeze
        self.reduce_patch_all = nn.ModuleList()
        self.decode_patch_all = nn.ModuleList()
        self.align_patch_len_all=nn.ModuleList()
        for i,patch_n in enumerate(self.configs.patch_num_all_scale) :
            if patch_n<=self.configs.threshold_patch_num or not self.configs.patch_squeeze:
                self.reduce_patch_all.append(nn.Linear(patch_n, patch_n // 1))
                self.decode_patch_all.append(
                    nn.Sequential(nn.Linear(patch_n // 1, patch_n // 1),
                                  nn.Linear(patch_n // 1, patch_n)))
            else:
                # print(self.configs.squeeze_factor[i])
                self.reduce_patch_all.append(nn.Linear(patch_n, patch_n // self.configs.squeeze_factor[i]))
                self.decode_patch_all.append(
                    nn.Sequential(nn.Linear(patch_n // self.configs.squeeze_factor[i], patch_n // self.configs.squeeze_factor[i]),
                                  nn.Linear(patch_n // self.configs.squeeze_factor[i], patch_n)))
            if self.configs.MAP:
                self.align_patch_len_all.append(nn.Sequential(nn.Linear(d_model, d_model // 2),
                                                     nn.Linear(d_model // 2, self.configs.patch_len_all_scale[i])))
            else:
                self.align_patch_len = nn.Sequential(nn.Linear(d_model, d_model // 2),
                                                         nn.Linear(d_model // 2, self.configs.patch_len_all_scale[i]))

        if self.individually_embed:
            self.W_P_all_scales=nn.ModuleList()
            self.W_Pos_all_scales = []
            for i,patch_n in enumerate(self.configs.patch_num_all_scale):
                if self.configs.MAP:
                    self.W_P_all_scales.append(nn.Linear(self.configs.equal_patch_len[i], d_model))
                    self.W_Pos_all_scales.append(positional_encoding(pe, learn_pe, patch_n, d_model).to(self.configs.device))
                else:

                    self.W_P_all_scales.append(nn.Linear(self.configs.max_patch_len, d_model))
                    self.W_Pos_all_scales.append(positional_encoding(pe, learn_pe, patch_n, d_model).to(self.configs.device))

        else:
            self.W_P = nn.Linear(self.configs.max_patch_len, d_model)
            self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        embed_dim=3
        if self.configs.patch_squeeze:
            self.patch_pj=nn.Linear(sum(self.configs.patch_num_all_scale_squeeze), embed_dim)
        else:
            self.patch_pj = nn.Linear(sum(self.configs.patch_num_all_scale), embed_dim)
    def forward(self, scale_all,x_patch_or=None,x_all_or=None) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        scale_all_rec={}
        scale_all_patch={} #used to calculate reconstruction loss

        if self.individually_embed:
            for scale, val in scale_all.items():
                scale_all[scale]=self.W_P_all_scales[scale](val.permute(0,1,3,2))
                scale_all[scale] = torch.reshape(scale_all[scale], (scale_all[scale].shape[0] * scale_all[scale].shape[1], scale_all[scale].shape[2], scale_all[scale].shape[3]))
                scale_all[scale] =self.dropout(scale_all[scale] + self.W_Pos_all_scales[scale])
                if self.configs.patch_squeeze:
                    if self.configs.patch_num_all_scale[scale]<=self.configs.threshold_patch_num:
                        continue
                    else:
                        scale_all_patch[scale] = val.view(val.shape[0] * val.shape[1],
                                                          val.shape[2], val.shape[3]).transpose(1, 2)
                        #PatchSqueeze
                        scale_all[scale] = self.reduce_patch_all[scale](scale_all[scale].transpose(1, 2)).transpose(1, 2)
                        scale_all_rec[scale] = self.decode_patch_all[scale](scale_all[scale].transpose(1, 2)).transpose(1, 2)
                        if self.configs.MAP:
                            scale_all_rec[scale] = self.align_patch_len_all[scale](scale_all_rec[scale])
                        else:
                            scale_all_rec[scale] = self.align_patch_len(scale_all_rec[scale])
                else:
                    scale_all_rec = None
                    scale_all_patch = None

            u = torch.cat(list(scale_all.values()), dim=-2)

        else:
            x=torch.cat(list(scale_all.values()),dim=-1)
            x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
            if self.configs.patch_pad:
                x = self.W_P(x)
            else:
                pass
            u = torch.reshape(x, (
            x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
            u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]
        if x_patch_or!=None:
            for k in x_patch_or.keys():
                x_patch_or[k]=x_patch_or[k].permute(0,1,3,2)

        forecasts=self.encoder(u,x_all_or=x_all_or)

        return forecasts,scale_all_rec,scale_all_patch



class LWI_Sub(nn.Module):
    """
    Pattern extraction for period gated attention module
    """
    def __init__(self, out_dim=None,configs=None):
        super().__init__()
        self.configs = configs
        self.thetas_dim=configs.seq_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.actv={'relu':F.relu,'sft':F.softplus,'tanh':F.tanh,'sigmoid':F.sigmoid,'sftmx':F.softmax}
        self.fc = nn.Sequential(nn.Linear(self.configs.seq_len, out_dim))
        self.hidden_sizes = [128, 256, 128]
        self.kernel_sizes = [9, 5, 3]
        self._build_model(configs.enc_in, self.hidden_sizes, self.kernel_sizes)

    def _build_model(self,input_size, hidden_sizes, kernel_sizes):
        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=hidden_sizes[0],
                               kernel_size=kernel_sizes[0])
        self.norm1 = nn.BatchNorm1d(num_features=hidden_sizes[0])
    def get_htensor(self, x):
        h = x
        h = f.pad(h, (int(self.kernel_sizes[0]/2), int(self.kernel_sizes[0]/2)), "constant", 0)
        h = self.norm1(self.conv1(h))
        return h

    def forward(self, x):

        h=self.get_htensor(x)
        h = F.max_pool1d(
            h.transpose(1, 2),
            kernel_size=128,
        ).transpose(1, 2)
        h = h.view(h.shape[0], h.shape[-1])
        return h

# Cell
class MLFEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, configs_=None,d_k=None, d_v=None, d_ff=None,
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.configs=configs_
        self.layers = nn.ModuleList([MLFEncoderLayer(d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,args=self.configs,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

        self.MLF=True
        self.flatten = nn.Flatten(start_dim=-2)
        self.enc_block_all=nn.ModuleList()
        for i_e in range(self.configs.e_layers):
            self.enc_block_all.append(nn.ModuleList())
        if self.MLF:
            for i_e in range(self.configs.e_layers):
                self.res_blocks_all = nn.ModuleList()
                for i in range(self.configs.scale_num):
                    self.enc_block_all[i_e].append(ForecastingBlock(self.configs.head_nf_res_all[-1],
                                                                    backcast_length=self.configs.head_nf_res_all[-1],
                                                                    forecast_length=self.configs.target_window,
                                                                    configs=self.configs))
        self.LWI_Sub = LWI_Sub( self.configs.scale_num, self.configs)
        self.forecast_all_dict = {}
        self.backcast_all_dict = {}
        self.attn_all_dict={}
        for _ in range(len(self.layers)):
            self.forecast_all_dict[_]={}
            self.backcast_all_dict[_]={}
            self.attn_all_dict[_]={}
        self.out_net2 = nn.Sequential(*[
            nn.Linear(self.configs.seq_len, self.configs.seq_len*2),
            nn.Tanh(),
            nn.Linear(self.configs.seq_len*2, len(self.configs.patch_num_all_scale)),
            nn.Tanh(),
        ])

        self.out_net1 = nn.Sequential(*[
            nn.Linear(self.configs.seq_len, self.configs.seq_len*2),
            nn.Tanh(),
            nn.Linear(self.configs.seq_len*2, len(self.configs.patch_num_all_scale)),
            nn.Tanh(),
        ])
        self.scaling = nn.Parameter(torch.tensor(self.configs.d_model ** -0.5), requires_grad=True)
    def forward(self, src:Tensor, x_all_or=None):
        output = src

        if self.configs.patch_squeeze:
            patch_num_all_scale=self.configs.patch_num_all_scale_squeeze
        else:
            patch_num_all_scale=self.configs.patch_num_all_scale

        for i_d,mod in enumerate(self.layers):
            self.configs.mod_i_d=i_d
            if i_d==0:
                z_all, scores = mod(output)
            else:
                z_all_at=torch.cat(list(z_all_next.values()),dim=-2)
                z_all,scores= mod(z_all_at)
            z_all_s = {}
            k = patch_num_all_scale[0]
            #scales split
            for i in range(self.configs.scale_num):
                z_all_s[i] = z_all[:, sum(patch_num_all_scale[:i]):sum(patch_num_all_scale[:i+1]), :]
                z_all_s[i] = torch.reshape(z_all_s[i], (-1, self.configs.n_var, z_all_s[i].shape[-2], z_all_s[i].shape[-1]))
                z_all_s[i] = z_all_s[i].permute(0, 1, 3, 2)
                if z_all_s[i].shape[-1] != patch_num_all_scale[-1] and not self.configs.MAP:
                    pad_size1 = patch_num_all_scale[-1] - z_all_s[i].size(-1)
                    z_all_s[i] = torch.nn.functional.pad(z_all_s[i], (0, pad_size1))
                k += patch_num_all_scale[i]
            patch_num, d_model = z_all_s[i].shape[-1], z_all_s[i].shape[-2]
            b_a = 0
            z_all_next={}
            z_all={}
            forecast_dict_temp={}
            for i in range(self.configs.scale_num):
                z_all_s[i] = self.flatten(z_all_s[i])
                if i != 0:
                    z_all[i] = z_all_s[i] - b_a
                else:
                    z_all[i] =z_all_s[i]
                z_all_next[i] = torch.reshape(z_all[i], (-1, patch_num_all_scale[-1], d_model))
                b, f = self.enc_block_all[i_d][i](z_all[i])
                forecast_dict_temp[i]=f
                self.forecast_all_dict[i_d][i]=f
                if self.configs.redundancy_scaling:
                    b_a += b * self.scaling
                else:
                    b_a += b
        enc_all = []
        for k,v in self.forecast_all_dict.items():
            f_all = []
            for k2, v2 in v.items():
                f_all.append(v2.unsqueeze(-2))
            forecast_all = torch.cat(f_all, dim=-2)
            enc_all.append(forecast_all.unsqueeze(0))

        forecast=torch.cat(enc_all,dim=0)
        forecast_encoder_level = torch.mean(forecast, dim=0).permute(0,1,3,2) #use sum or mean to aggregat forecasts of encoders
        #strategy
        if self.configs.LWI:
            h = self.LWI_Sub(x_all_or[-1])
            gate_m=self.out_net1(h)*self.out_net2(h)
            gate_m=F.sigmoid(gate_m)
            gate_m=gate_m.unsqueeze(1).unsqueeze(1).repeat(1,self.configs.enc_in,self.configs.pred_len,1)
            adaptively_selected_forecast = forecast_encoder_level*gate_m
            adaptively_selected_forecast=torch.mean(adaptively_selected_forecast,dim=-1)
        else:
            adaptively_selected_forecast = torch.mean(forecast_encoder_level, dim=-1)

        return adaptively_selected_forecast


class MLFEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,args=None,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.configs=args
        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        print('n_heads',n_heads)
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)
        if self.res_attention:
            return src, scores
        else:
            return src

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        # print(Q.shape,K.shape,V.shape)

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]

        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.scale2 = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=True)

        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

