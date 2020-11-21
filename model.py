class PredRNN(nn.Module):
    def __init__(self, num_layers, num_h):
        super(PredRNN, self).__init__()
        self.num_layers = num_layers
        self.num_h = num_h
        
        self.shape = []
        
        cell_list = []

        for i in range(self.num_layers):
            if i == 0:
                num_hidden_in = 1
            else:
                num_hidden_in = self.num_h[i-1]

            cell_list.append(
                SpatioTemporalLSTMCell(num_hidden_in, num_h[i], 120, 120, 3)
            )

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_h[-1], 1, 1, 1, 0)

    def forward(self, img):
        batch, length, channel, width, height = img.shape
        
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_h[i], height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        mem = torch.zeros([batch, self.num_h[0], height, width]).to(device)

        for t in range(5):
            if t < 4:
                net = img[:, t]

            h_t[0], c_t[0], mem = self.cell_list[0](net, h_t[0], c_t[0], mem)

            for j in range(2, self.num_layers):
                h_t[j], c_t[j], mem = self.cell_list[i](h_t[i-1], h_t[i], c_t[i], mem)

            last_result = self.conv_last(h_t[self.num_layers - 1])

        return last_result
