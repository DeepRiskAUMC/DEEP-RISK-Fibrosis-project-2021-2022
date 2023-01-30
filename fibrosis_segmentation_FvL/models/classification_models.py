import torch
import torch.nn as nn
from models.densenet import DenseNet as DenseNet3D

class Simple_CNN_old(nn.Module):
    def __init__(self, in_channels, mid_channels=[32,64,128,256], n_classes=1, sigmoid_finish=True):
        super(Simple_CNN_old, self).__init__()
        self.in_channels = in_channels
        self.last_dimension = mid_channels[-1]
        self.sigmoid_finish = sigmoid_finish

        conv_list = []
        
        for i in range(len(mid_channels)):
            if i == 0:
                conv_list.append(nn.Conv3d(in_channels, mid_channels[i], kernel_size=3, padding=1))
            else:
                conv_list.append(nn.Conv3d(mid_channels[i-1], mid_channels[i], kernel_size=3, padding=1))
            conv_list.append(nn.BatchNorm3d(mid_channels[i]))
            conv_list.append(nn.ReLU(inplace=True))
        
        self.conv_layers = nn.Sequential(*conv_list)
        self.linear = nn.Linear(mid_channels[-1], n_classes)
        if self.sigmoid_finish:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        batch_size, channels = x.shape[:2]
        x = x.view(batch_size, channels, -1)
        x = x.mean(dim=-1)
        x = self.linear(x)
        if self.sigmoid_finish:
            x = self.sigmoid(x)
        return x

# class Simple_CNN(nn.Module):
#     def __init__(self, in_channels, cnn_channels=[32,32], lin_channels=[64], n_classes=1, sigmoid_finish=True):
#         super(Simple_CNN, self).__init__()
#         self.in_channels = in_channels
#         self.last_dimension = cnn_channels[-1]
#         self.sigmoid_finish = sigmoid_finish

#         conv_list = []
        
#         for i in range(len(cnn_channels)):
#             if i == 0:
#                 conv_list.append(nn.Conv3d(in_channels, cnn_channels[i], kernel_size=3, padding=1))
#             else:
#                 conv_list.append(nn.Conv3d(cnn_channels[i-1], cnn_channels[i], kernel_size=3, padding=1))
#             conv_list.append(nn.BatchNorm3d(cnn_channels[i]))
#             conv_list.append(nn.ReLU(inplace=True))
        
#         self.conv_layers = nn.Sequential(*conv_list)

#         linear_list = []
#         for i in range(len(lin_channels)):
#             if i == 0:
#                 linear_list.append(nn.Linear(cnn_channels[-1], lin_channels[i]))
#             else:
#                 linear_list.append(nn.Linear(lin_channels[i-1], lin_channels[i]))
#             linear_list.append(nn.ReLU(inplace=True))
#         self.linear_layers = nn.Sequential(*linear_list)
#         self.final_linear = nn.Linear(lin_channels[-1], n_classes)
#         print('sigmoid_finish inside CNN', self.sigmoid_finish)
#         if self.sigmoid_finish:
#             self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.conv_layers(x)
#         batch_size, channels = x.shape[:2]
#         x = x.view(batch_size, channels, -1)
#         x = x.mean(dim=-1)
#         x = self.linear_layers(x)
#         x = self.final_linear(x)
#         if self.sigmoid_finish:
#             x = self.sigmoid(x)
#         return x

class Simple_CNN_clinical(nn.Module):
    def __init__(self, in_channels, cnn_channels=[16,16], lin_channels=[64], n_classes=1, sigmoid_finish=True, use_MRI_features=False):
        super(Simple_CNN_clinical, self).__init__()
        self.in_channels = in_channels
        self.last_dimension = cnn_channels[-1]
        self.sigmoid_finish = sigmoid_finish
        self.use_MRI_features = use_MRI_features

        conv_list = []
        
        for i in range(len(cnn_channels)):
            if i == 0:
                conv_list.append(nn.Conv3d(in_channels, cnn_channels[i], kernel_size=3, padding=1))
            else:
                conv_list.append(nn.Conv3d(cnn_channels[i-1], cnn_channels[i], kernel_size=3, padding=1))
            conv_list.append(nn.BatchNorm3d(cnn_channels[i]))
            conv_list.append(nn.ReLU(inplace=True))
        
        self.conv_layers = nn.Sequential(*conv_list)

        linear_list = []
        if self.use_MRI_features:
            input_linear = cnn_channels[-1] + 8
        else:
            input_linear = cnn_channels[-1]
        for i in range(len(lin_channels)):
            if i == 0:
                linear_list.append(nn.Linear(input_linear, lin_channels[i]))
            else:
                linear_list.append(nn.Linear(lin_channels[i-1], lin_channels[i]))
            linear_list.append(nn.ReLU(inplace=True))
        self.linear_layers = nn.Sequential(*linear_list)
        self.final_linear = nn.Linear(lin_channels[-1], n_classes)
        print('sigmoid_finish inside CNN', self.sigmoid_finish)
        if self.sigmoid_finish:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, clinical):
        # x, clinical = x
        x = self.conv_layers(x)
        batch_size, channels = x.shape[:2]
        x = x.view(batch_size, channels, -1)
        x = x.mean(dim=-1)
        if self.use_MRI_features:
            x = torch.cat((x, clinical), dim=1)
        x = self.linear_layers(x)
        x = self.final_linear(x)
        if self.sigmoid_finish:
            x = self.sigmoid(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, lin_channels=[64], n_classes=1, sigmoid_finish=True, use_MRI_features=False, flatten_or_maxpool='maxpool') -> None:
        super().__init__()
        if flatten_or_maxpool == 'flatten':
            self.flatten = True
        else:
            self.flatten = False
        if self.flatten:
            print('flattening predictions')
            if use_MRI_features:
                in_channels = (in_channels - 8) * 832 + 8
            else:
                in_channels = in_channels * 832
        self.sigmoid_finish = sigmoid_finish
        self.use_MRI_features = use_MRI_features
        linear_list = []
        for i in range(len(lin_channels)):
            if i == 0:
                linear_list.append(nn.Linear(in_channels, lin_channels[i]))
            else:
                linear_list.append(nn.Linear(lin_channels[i-1], lin_channels[i]))
            linear_list.append(nn.ReLU(inplace=True))
        linear_list.append(nn.Linear(lin_channels[-1], n_classes))
        self.linear_layers = nn.Sequential(*linear_list)
        if self.sigmoid_finish:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, mri_features):
        if not self.flatten:
            #global max pooling
            x = torch.amax(x.view(x.shape[0], x.shape[1], -1), dim=2)
        else:
            # flatten
            x = x.view(x.shape[0], -1)
        if self.use_MRI_features:
            x = torch.cat((x, mri_features), dim=1)
        x = self.linear_layers(x)
        if self.sigmoid_finish:
            x = self.sigmoid(x)
        return x

# class DenseNetMaxpoolClassificationModel(nn.Module):
#     def __init__(self, in_channels=1, n_classes=1, dropout=0, sigmoid_finish=False) -> None:
#         super().__init__()
#         self.sigmoid_finish = sigmoid_finish
#         self.model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
#         self.model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         self.output_features = 1024
#         self.model.classifier = nn.Linear(self.output_features, n_classes)

#     def forward(self, x):
#         batch_size, _, slice_count, height, width = x.shape
#         max_pool_outputs = torch.zeros(batch_size, 1, requires_grad=True).to(x.device)
#         # print('max_pool_outputs device', max_pool_outputs.device, max_pool_outputs.requires_grad)
#         for j in range(batch_size):
#             projected_slices = []
#             for i in range(slice_count):
#                 slice = x[j,:,i,:,:].view(1, 1, height, width)
#                 if torch.any(slice > 0):
#                     output = self.model(slice)
#                     projected_slices.append(output)
#             projected_slices = torch.tensor(projected_slices).view(1, -1).to(x.device)
#             # print('shape projected_slices:', projected_slices.shape)
#             max_pool_output = torch.amax(projected_slices, dim=-1)
#             max_pool_outputs[j] = max_pool_output
#         if self.sigmoid_finish:
#             x = nn.Sigmoid()(max_pool_outputs)
#         else:
#             x = max_pool_outputs
#         return x

class DenseNetPaddingClassificationModel(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, dropout=0.1, sigmoid_finish=False) -> None:
        super().__init__()
        self.sigmoid_finish = sigmoid_finish
        self.model = DenseNet3D(n_input_channels=in_channels, drop_rate=dropout, num_classes=n_classes, no_max_pool=True)

    def forward(self, x):
        batch_size, _, slice_count, height, width = x.shape
        x = self.model(x)
        if self.sigmoid_finish:
            x = nn.Sigmoid()(x)
        else:
            x = x
        return x

    def remove_classifier(self):
        self.model.remove_classifier()

class MultiInputDenseNetClassificationModel(nn.Module):
    def __init__(self, myocard_model, fibrosis_model, densenet_model, sigmoid_finish=False) -> None:
        super().__init__()
        self.myocard_model = myocard_model
        self.fibrosis_model = fibrosis_model
        self.densenet_model = densenet_model
        self.fibrosis_fc1 = nn.Linear(512, 32)
        self.fibrosis_fc2 = nn.Linear(256, 32)
        self.fibrosis_fc3 = nn.Linear(128, 32)
        self.fibrosis_fc4 = nn.Linear(64, 32)
        
        self.densenet_fc = nn.Linear(1024, 32)

        self.final_fc = nn.Linear(200, 1)
        
        self.sigmoid_finish = sigmoid_finish
        if self.sigmoid_finish:
            self.sigmoid = nn.Sigmoid()

    def forward(self, LGE_img, MRI_features):
        device = LGE_img.device
        batch_size, _, slice_count, height, width = LGE_img.shape

        segment_output1 = torch.zeros((batch_size, 512, slice_count, 8, 8), requires_grad=True).to(device)
        segment_output2 = torch.zeros((batch_size, 256, slice_count, 16, 16), requires_grad=True).to(device)
        segment_output3 = torch.zeros((batch_size, 128, slice_count, 33, 33), requires_grad=True).to(device)
        segment_output4 = torch.zeros((batch_size, 64, slice_count, 66, 66), requires_grad=True).to(device)
        segment_output5 = torch.zeros((batch_size, 32, slice_count, 132, 132), requires_grad=True).to(device)

        for i in range(slice_count):
            slice = LGE_img[:,:,i,:,:].view(batch_size, 1, height, width)
            if torch.any(slice > 0):
                myo_pred = self.myocard_model(slice)
                stacked_input = torch.stack([slice, myo_pred], dim=1).squeeze()
                output = self.fibrosis_model(stacked_input)
                segment_output1[:,:,i,:,:] = output[0]
                segment_output2[:,:,i,:,:] = output[1]
                segment_output3[:,:,i,:,:] = output[2]
                segment_output4[:,:,i,:,:] = output[3]
                segment_output5[:,:,i,:,:] = output[4]
        fib_out1 = segment_output1.view(batch_size, 512, -1)
        fib_out1 = torch.amax(fib_out1, dim=2)
        fib_out2 = segment_output2.view(batch_size, 256, -1)
        fib_out2 = torch.amax(fib_out2, dim=2)
        fib_out3 = segment_output3.view(batch_size, 128, -1)
        fib_out3 = torch.amax(fib_out3, dim=2)
        fib_out4 = segment_output4.view(batch_size, 64, -1)
        fib_out4 = torch.amax(fib_out4, dim=2)
        fib_out5 = segment_output5.view(batch_size, 32, -1)
        fib_out5 = torch.amax(fib_out5, dim=2)
        fib_out1 = self.fibrosis_fc1(fib_out1)
        fib_out2 = self.fibrosis_fc2(fib_out2)
        fib_out3 = self.fibrosis_fc3(fib_out3)
        fib_out4 = self.fibrosis_fc4(fib_out4)

        densenet_out = self.densenet_model(LGE_img)
        densenet_out = self.densenet_fc(densenet_out)
        cat_vector = torch.cat([MRI_features, fib_out1, fib_out2, fib_out3, fib_out4, fib_out5, densenet_out], dim=1)

        output = self.final_fc(cat_vector)
        if self.sigmoid_finish:
            output = self.sigmoid(output)
        return output
