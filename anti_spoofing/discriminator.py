import torch
import torch.nn as nn
import torch.nn.functional as F

class highwayConv(nn.Module):
    """
    This is a highway convolution layer which is used in some models.
    It will use "same" convolution.
    """

    def __init__(self, dimension, kernel_size, dilation, causal=False):
        """
        Args:
        --dimension. Number of input and output channels. They always keep same in highway nets.
        --causal. A boolean on whether to use causal convolution or ordinary one.
        """
        super(highwayConv, self).__init__()
        self.dimension = dimension
        self.causal = causal

        # Same Convolution. Stride = 1.
        # L_out = L_in + 2*padding - dilation*(kernel_size - 1)
        # As in this model, kernel_size is always odd,
        # we do not consider the case that pad is not an integer.
        self.pad = dilation*(kernel_size-1) // 2

        self.conv = nn.Conv1d(in_channels=dimension, out_channels=2*dimension, kernel_size=kernel_size, padding=0 if causal else self.pad, dilation=dilation)
        self.ln1 = nn.LayerNorm(normalized_shape=dimension)
        self.ln2 = nn.LayerNorm(normalized_shape=dimension)
        self.dp = nn.Dropout(p=0.05)

    def forward(self, inputs):
        """
        --inputs: (Batch, dimension, timeseries)
        --x: (Batch, 2*dimension, timeseries)
        --H1/H2: (Batch, dimension, timeseries)
        --outputs: (Batch, dimension, timeseries)
        """

        # zero-padding prior to the inputs to ensure causal convolution.
        if self.causal and self.pad>0:
            d1, d2, d3 = inputs.size()
            inputs = torch.cat((torch.zeros((d1, d2, 2*self.pad)).to(device), inputs), dim=-1)

        #print('hc inputs device: ', inputs.device)

        x = self.conv(inputs)
        H1 = x[:, :self.dimension, :]
        H2 = x[:, self.dimension:, :]
        H1 = self.ln1(H1.permute(0,2,1)).permute(0,2,1)
        H2 = self.ln2(H2.permute(0,2,1)).permute(0,2,1)
        outputs = F.sigmoid(H1)*H2+(1-F.sigmoid(H1))*inputs[:, :, 2*self.pad if self.causal else 0:]
        outputs = self.dp(outputs)
        return outputs

class melDisc(nn.Module):
    def __init__(self, freq_bins, disc_dim):
        super(melDisc, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=freq_bins, out_channels=disc_dim, kernel_size=1)
        self.ln1 = nn.LayerNorm(normalized_shape=disc_dim)
        self.dp1 = nn.Dropout(p=0.05)
        self.hc = highwayConv(dimension=disc_dim, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=disc_dim, out_channels=64, kernel_size=1)
        self.pl1 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(normalized_shape=64)
        self.dp2 = nn.Dropout(p=0.05)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1)
        self.pl2 = nn.AvgPool1d(kernel_size=2)
        self.ln3 = nn.LayerNorm(normalized_shape=16)

        self.conv4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.ln4 = nn.LayerNorm(normalized_shape=4)
        self.conv5 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1)
        self.pl3 = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.ln1(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp1(x)
        x = self.hc(x)
        x = self.conv2(x)
        x = self.pl1(x)
        x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp2(F.leaky_relu(x, 0.05))
        x = self.conv3(x)
        x = self.pl2(x)
        x = self.ln3(x.permute(0,2,1)).permute(0,2,1)

        x = self.conv4(F.leaky_relu(x, 0.05))
        x = self.ln4(x.permute(0,2,1)).permute(0,2,1)
        x = self.conv5(F.leaky_relu(x, 0.05))
        x = self.pl3(x)
        x = F.sigmoid(x)
        return x

class linDisc(nn.Module):
    def __init__(self, freq_bins, disc_dim):
        super(linDisc, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=freq_bins, out_channels=disc_dim, kernel_size=1)
        self.ln1 = nn.LayerNorm(normalized_shape=disc_dim)
        self.dp1 = nn.Dropout(p=0.05)
        self.hc = highwayConv(dimension=disc_dim, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=disc_dim, out_channels=64, kernel_size=1)
        self.pl1 = nn.AvgPool1d(kernel_size=8)
        self.ln2 = nn.LayerNorm(normalized_shape=64)
        self.dp2 = nn.Dropout(p=0.05)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1)
        self.pl2 = nn.AvgPool1d(kernel_size=4)
        self.ln3 = nn.LayerNorm(normalized_shape=16)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.ln4 = nn.LayerNorm(normalized_shape=8)
        self.conv5 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1)
        self.pl3 = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.ln1(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp1(x)
        x = self.hc(x)
        x = self.conv2(x)
        x = self.pl1(x)
        x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp2(F.leaky_relu(x, 0.05))
        x = self.conv3(x)
        x = self.pl2(x)
        x = self.ln3(x.permute(0,2,1)).permute(0,2,1)
        x = self.conv4(F.leaky_relu(x, 0.05))
        x = self.ln4(x.permute(0,2,1)).permute(0,2,1)
        x = self.conv5(F.leaky_relu(x, 0.05))
        x = self.pl3(x)
        x = F.sigmoid(x)
        return x

class melDisc_v1(nn.Module):
    def __init__(self, freq_bins, disc_dim):
        super(melDisc, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=freq_bins, out_channels=disc_dim, kernel_size=1)
        self.ln1 = nn.LayerNorm(normalized_shape=disc_dim)
        self.dp1 = nn.Dropout(p=0.05)
        self.hc = highwayConv(dimension=disc_dim, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=disc_dim, out_channels=64, kernel_size=1)
        self.pl1 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(normalized_shape=64)
        self.dp2 = nn.Dropout(p=0.05)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1)
        # self.pl2 = nn.AvgPool1d(kernel_size=2)
        self.ln3 = nn.LayerNorm(normalized_shape=16)

        self.conv4 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=1)
        self.ln4 = nn.LayerNorm(normalized_shape=4)
        self.conv5 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.pl3 = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.ln1(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp1(x)
        x = self.hc(x)
        x = self.conv2(x)
        x = self.pl1(x)
        x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp2(F.leaky_relu(x, 0.05))
        x = self.conv3(x)
        # x = self.pl2(x)
        x = self.ln3(x.permute(0,2,1)).permute(0,2,1)

        x = self.conv4(F.leaky_relu(x, 0.05))
        x = self.ln4(x.permute(0,2,1)).permute(0,2,1)
        x = self.conv5(F.leaky_relu(x, 0.05))
        x = self.pl3(x)
        x = F.sigmoid(x)
        return x

class linDisc_v1(nn.Module):
    def __init__(self, freq_bins, disc_dim):
        super(linDisc, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=freq_bins, out_channels=disc_dim, kernel_size=1)
        self.ln1 = nn.LayerNorm(normalized_shape=disc_dim)
        self.dp1 = nn.Dropout(p=0.05)
        self.hc = highwayConv(dimension=disc_dim, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=disc_dim, out_channels=64, kernel_size=1)
        self.pl1 = nn.AvgPool1d(kernel_size=8)
        self.ln2 = nn.LayerNorm(normalized_shape=64)
        self.dp2 = nn.Dropout(p=0.05)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1)
        # self.pl2 = nn.AvgPool1d(kernel_size=4)
        self.ln3 = nn.LayerNorm(normalized_shape=16)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.ln4 = nn.LayerNorm(normalized_shape=8)
        self.conv5 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1)
        self.pl3 = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.ln1(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp1(x)
        x = self.hc(x)
        x = self.conv2(x)
        x = self.pl1(x)
        x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp2(F.leaky_relu(x, 0.05))
        x = self.conv3(x)
        # x = self.pl2(x)
        x = self.ln3(x.permute(0,2,1)).permute(0,2,1)
        x = self.conv4(F.leaky_relu(x, 0.05))
        x = self.ln4(x.permute(0,2,1)).permute(0,2,1)
        x = self.conv5(F.leaky_relu(x, 0.05))
        x = self.pl3(x)
        x = F.sigmoid(x)
        return x

class melDisc_v2(nn.Module):
    def __init__(self, freq_bins, disc_dim):
        super(melDisc, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=freq_bins, out_channels=disc_dim, kernel_size=1)
        self.ln1 = nn.LayerNorm(normalized_shape=disc_dim)
        self.dp1 = nn.Dropout(p=0.05)
        self.hc = highwayConv(dimension=disc_dim, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=disc_dim, out_channels=64, kernel_size=1)
        self.pl1 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(normalized_shape=64)
        self.dp2 = nn.Dropout(p=0.05)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1)
        self.pl2 = nn.AvgPool1d(kernel_size=2)
        self.ln3 = nn.LayerNorm(normalized_shape=16)

        self.conv3_2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.pl2_2 = nn.AvgPool1d(kernel_size=2)
        self.ln3_2 = nn.LayerNorm(normalized_shape=8)

        self.conv4 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=1)
        self.ln4 = nn.LayerNorm(normalized_shape=4)
        self.conv5 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.pl3 = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.ln1(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp1(x)
        x = self.hc(x)
        x = self.conv2(x)
        x = self.pl1(x)
        x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp2(F.leaky_relu(x, 0.05))
        x = self.conv3(x)
        x = self.pl2(x)
        x = self.ln3(x.permute(0,2,1)).permute(0,2,1)

        x = self.conv3_2(F.leaky_relu(x, 0.05))
        x = self.pl2_2(x)
        x = self.ln3_2(x.permute(0,2,1)).permute(0,2,1)

        x = self.conv4(F.leaky_relu(x, 0.05))
        x = self.ln4(x.permute(0,2,1)).permute(0,2,1)
        x = self.conv5(F.leaky_relu(x, 0.05))
        x = self.pl3(x)
        x = F.sigmoid(x)
        return x

class linDisc_v2(nn.Module):
    def __init__(self, freq_bins, disc_dim):
        super(linDisc, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=freq_bins, out_channels=disc_dim, kernel_size=1)
        self.ln1 = nn.LayerNorm(normalized_shape=disc_dim)
        self.dp1 = nn.Dropout(p=0.05)
        self.hc = highwayConv(dimension=disc_dim, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=disc_dim, out_channels=64, kernel_size=1)
        self.pl1 = nn.AvgPool1d(kernel_size=8)
        self.ln2 = nn.LayerNorm(normalized_shape=64)
        self.dp2 = nn.Dropout(p=0.05)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1)
        self.pl2 = nn.AvgPool1d(kernel_size=4)
        self.ln3 = nn.LayerNorm(normalized_shape=16)

        self.conv3_2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1)
        self.pl2_2 = nn.AvgPool1d(kernel_size=2)
        self.ln3_2 = nn.LayerNorm(normalized_shape=8)

        self.conv4 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=1)
        self.ln4 = nn.LayerNorm(normalized_shape=8)
        self.conv5 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.pl3 = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.ln1(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp1(x)
        x = self.hc(x)
        x = self.conv2(x)
        x = self.pl1(x)
        x = self.ln2(x.permute(0,2,1)).permute(0,2,1)
        x = self.dp2(F.leaky_relu(x, 0.05))
        x = self.conv3(x)
        x = self.pl2(x)
        x = self.ln3(x.permute(0,2,1)).permute(0,2,1)

        x = self.conv3_2(F.leaky_relu(x, 0.05))
        x = self.pl2_2(x)
        x = self.ln3_2(x.permute(0,2,1)).permute(0,2,1)

        x = self.conv4(F.leaky_relu(x, 0.05))
        x = self.ln4(x.permute(0,2,1)).permute(0,2,1)
        x = self.conv5(F.leaky_relu(x, 0.05))
        x = self.pl3(x)
        x = F.sigmoid(x)
        return x

# def conv3x3(planes):
#     ''' 3x3 convolution '''
#     return nn.Conv2d(planes, planes, kernel_size=(3,3), padding=(1,1), bias=False)

# class ResBasicBlock(nn.Module):
#     ''' basic Conv2D Block for ResNet '''
#     def __init__(self, planes):

#         super(ResBasicBlock, self).__init__()
        
#         self.bn1  = nn.BatchNorm2d(planes)
#         self.re1  = nn.LeakyReLU(0.05, inplace=True)
#         self.cnn1 = conv3x3(planes)
#         self.bn2  = nn.BatchNorm2d(planes)
#         self.re2  = nn.LeakyReLU(0.05, inplace=True)
#         self.cnn2 = conv3x3(planes)

#     def forward(self, x):
#         residual = x
#         x = self.cnn2(self.re2(self.bn2(self.cnn1(self.re1(self.bn1(x))))))
#         x += residual 
        
#         return x

# class DRS(nn.Module):
#     ''' small ResNet (less GPU memory) for 257 by 400 feature map '''
#     def __init__(self, num_classes, resnet_blocks=1, focal_loss=False):

#         super(DRS, self).__init__()
        
#         self.focal_loss = focal_loss

#         self.expansion = nn.Conv2d(1, 8, kernel_size=(3,3), padding=(1,1))
#         ## block 1
#         layers = []
#         for i in range(resnet_blocks):
#             layers.append(ResBasicBlock(8))
#         self.block1 = nn.Sequential(*layers)
#         self.mp1    = nn.AvgPool2d(kernel_size=(2,2))
#         self.cnn1   = nn.Conv2d(8, 16, kernel_size=(3,3), dilation=(2,2))
#         ## block 2
#         layers = []
#         for i in range(resnet_blocks):
#             layers.append(ResBasicBlock(16))
#         self.block2 = nn.Sequential(*layers)
#         self.mp2    = nn.AvgPool2d(kernel_size=(2,2))
#         self.cnn2   = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(4,4))
#         ## block 3
#         layers = []
#         for i in range(resnet_blocks):
#             layers.append(ResBasicBlock(32))
#         self.block3 = nn.Sequential(*layers)
#         self.mp3    = nn.AvgPool2d(kernel_size=(2,2))
#         self.cnn3   = nn.Conv2d(32, 64, kernel_size=(3,3), dilation=(8,8))
#         ## block 4
#         layers = []
#         for i in range(resnet_blocks):
#             layers.append(ResBasicBlock(64))
#         self.block4 = nn.Sequential(*layers)
#         self.mp4    = nn.AvgPool2d(kernel_size=(2,2))
#         self.cnn4   = nn.Conv2d(64, 64, kernel_size=(3,3), dilation=(9,6)) 

#         self.flat_feats = 64*3*2

#         self.fc  = nn.Linear(self.flat_feats, 100)
#         self.bn  = nn.BatchNorm1d(100)    
#         self.re  = nn.LeakyReLU(0.05, inplace=True)
#         self.fc_out  = nn.Linear(100, num_classes)
    
#         ## Weights initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d or nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
    
#     def forward(self, x):
#         x = self.expansion(x)
#         ## block 1
#         x = self.cnn1(self.mp1(self.block1(x)))
#         #print(x.size())
#         ## block 2
#         x = self.cnn2(self.mp2(self.block2(x)))
#         #print(x.size())
#         ## block 3
#         x = self.cnn3(self.mp3(self.block3(x)))
#         #print(x.size())
#         ## block 4
#         x = self.cnn4(self.mp4(self.block4(x)))
#         #print(x.size())
#         ## FC
#         x = self.fc_out(self.re(self.bn(self.fc(x.view(-1, self.flat_feats)))))
#         #print(x.size())
 
#         if self.focal_loss: return x
#         else: return F.softmax(x, dim=-1) # take log-softmax over C classes