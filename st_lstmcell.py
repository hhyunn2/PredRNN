import numpy as np

import torch_
import torch.nn as nn
import torch.utils.data


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, w, h, kernel_size):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        

        # Biases
        self.bias_f = 1.0
        self.bias_g = 0.0
        self.bias_o = 0.0
        self.bias_i = 0.0
        
        
        # Convolutional Layers
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size, 1, 1),
            nn.LayerNorm([num_hidden * 7, w, h])
        )

        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size, 1, 1),
            nn.LayerNorm([num_hidden * 4, w, h])
        )

        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size, 1, 1),
            nn.LayerNorm([num_hidden * 3, w, h])
        )
        
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size, 1, 1),
            nn.LayerNorm([num_hidden, w, h])
        )

        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, 1, 1, 0)


    def forward(self, x_t, h_t, c_t, m_t):
        cc_x = self.conv_x(x_t)
        cc_h = self.conv_h(h_t)
        cc_m = self.conv_m(m_t)

        i_x, f_x, g_x, o_x, i_x2, f_x2, g_x2 = torch.split(cc_x, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(cc_h, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(cc_m, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + self.bias_i)
        f_t = torch.sigmoid(f_x + f_h + self.bias_f)
        g_t = torch.tanh(g_x + g_h + self.bias_g)

        c_new = f_t * c_t + i_t * g_t

        f_t2 = torch.sigmoid(f_x2 + f_m + self.bias_f)
        i_t2 = torch.sigmoid(i_x2 + i_m)
        
        g_t2 = torch.tanh(g_x2 + g_m)

        m_new = f_t2 * m_t + i_t2 * g_t2

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem) + self.bias_o)
        mem = self.conv_last(mem)
        h_new = o_t * torch.tanh(mem)

        return h_new, c_new, m_new
