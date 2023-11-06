# created 120923
# last edited 120923

# py file containing model architectures

# Imports
import torch
import torch.nn as nn

# Define model - copy of what worked on MNIST
def vgg16net(in_chan, f_lin_lay, l_lin_lay, ks, dropout):
    class vgg16TorchNet(nn.Module):
        # 7 conv layers, 3 linear
        def __init__(self):
            super(vgg16TorchNet, self).__init__()
            self.flatten = nn.Flatten()

            self.conv_layers = nn.Sequential(  # 1, 2, 144, 452
                  nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=ks, padding=2),
                  nn.ReLU(), #inplace=True
                  nn.Dropout(p=0.5),
                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks, padding=2),
                  nn.ReLU(), #inplace=True
                  nn.Conv2d(in_channels =64, out_channels=64, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2, 2),
                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=ks, padding=2),
                  nn.ReLU(), #inplace=True
                  nn.Conv2d(in_channels =128, out_channels=128, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=ks, padding=2),
                  nn.ReLU(), #inplace=True
                  nn.Conv2d(in_channels =256, out_channels=256, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Dropout(p=0.5), # (1x258048 and 16384x100)
              )

            self.linear_1 = nn.Sequential(    #1x16384 and 4096x100)
                nn.Linear(f_lin_lay, 100),
                nn.ReLU(),
                nn.Linear(100,100),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(100,l_lin_lay),
                nn.Softmax(),
            )

        def forward(self, x):
          #forward method. opposition to backward pass
          #print(x.shape)
          x= self.conv_layers(x)
          x = x.flatten()
          x = x.squeeze()
          #print('conv x', x.shape)
          x = self.linear_1(x)
          #print('lin1 x', x)
          return x
    model = vgg16TorchNet()
    return model

    
# Editable network
def build_net(lin_layer_size, dropout, first_lin_lay, ks, in_chan, pad_size):

    """class AddPadding(nn.Module):
        def __init__(self, pad_size):
            super(AddPadding,self).__init__()
        def forward(self, x):
            # add padding to tensor image
            img = x.squeeze()
            
            # select padding from sides of image
            left_x = img[:,:,:pad_size]
            right_x = img[:,:,-pad_size:]
            
            # get sizes for new image
            _y = img.shape[1]
            _x = img.shape[2]+(pad_size*2)
            
            # create empty array for new image size
            new_x = torch.zeros((3, _y, _x))
            # fill empty array
            new_x[:,:,:pad_size] = right_x
            new_x[:,:,pad_size:-pad_size] = img
            new_x[:,:,-pad_size:] = left_x
            
            # convert to tensor
            new_x = torch.tensor(new_x, dtype=torch.float32)
            new_x = torch.unsqueeze(new_x, 0)
            return new_x"""



    class EditNet(nn.Module):
        # 6 conv layers, 3 linear
        def __init__(self):
            super(EditNet, self).__init__()
            self.flatten = nn.Flatten()

            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=ks),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                #nn.Lambda(Lambda x: self._pad(x,2)),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks, padding=2),
                nn.ReLU(), #inplace=True
                nn.Conv2d(in_channels =64, out_channels=64, kernel_size=ks),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=ks, padding=2),
                nn.ReLU(), #inplace=True
                nn.Conv2d(in_channels =128, out_channels=128, kernel_size=ks),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=ks, padding=2),
                nn.ReLU(), #inplace=True
                nn.Conv2d(in_channels =256, out_channels=256, kernel_size=ks),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Dropout(p=dropout), # (1x258048 and 16384x100)
			)

            self.linear_1 = nn.Sequential(    #1x16384 and 4096x100)
                nn.Linear(first_lin_lay, lin_layer_size),
                nn.ReLU(),
                nn.Linear(lin_layer_size,lin_layer_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(lin_layer_size,11),
                nn.Softmax(),
			)

        

        def forward(self, x):
            x= self.conv_layers(x)
            x = x.flatten()
            x = x.squeeze()
            x = self.linear_1(x)
            return x
    model = EditNet()
    return model

def smallnet1(in_chan, f_lin_lay, l_lin_lay, ks):
    class SmallNet1(nn.Module):
        # 4 conv layers, 3 linear
        def __init__(self):
            super(SmallNet1, self).__init__()
            self.flatten = nn.Flatten()

            self.conv_layers = nn.Sequential(  # 1, 2, 144, 452
                  nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=ks, padding=2),
                  nn.ReLU(), #inplace=True
                  nn.Dropout(p=0.5),
                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks, padding=2),
                  nn.ReLU(), #inplace=True
                  nn.Conv2d(in_channels =64, out_channels=64, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2, 2),
                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=ks, padding=2),
                  nn.ReLU(), #inplace=True
                  nn.MaxPool2d(2,2),
                  nn.Dropout(p=0.5), # (1x258048 and 16384x100)
              )

            self.linear_1 = nn.Sequential(    #1x16384 and 4096x100)
                nn.Linear(f_lin_lay, 100),
                nn.ReLU(),
                nn.Linear(100,100),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(100,l_lin_lay),
                nn.Softmax(),
            )

        def forward(self, x):
          x= self.conv_layers(x)
          x = x.flatten()
          x = x.squeeze()
          x = self.linear_1(x)
          return x
    model = SmallNet1()
    return model






def smallnet2(in_chan, f_lin_lay, l_lin_lay, ks):
    class SmallNet2(nn.Module):
        # 3 conv layers, 2 linear
        def __init__(self):
            super(SmallNet2, self).__init__()
            self.flatten = nn.Flatten()

            self.conv_layers = nn.Sequential(  # 1, 2, 144, 452
                  nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=ks, padding=2),
                  nn.ReLU(), #inplace=True
                  nn.Dropout(p=0.5),
                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks, padding=2),
                  nn.ReLU(), #inplace=True
                  nn.Conv2d(in_channels =64, out_channels=64, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Dropout(p=0.5), # (1x258048 and 16384x100)
              )

            self.linear_1 = nn.Sequential(    #1x16384 and 4096x100)
                nn.Linear(f_lin_lay, 100),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(100,l_lin_lay),
                nn.Softmax(),
            )


        def forward(self, x):
          x= self.conv_layers(x)
          x = x.flatten()
          x = x.squeeze()
          x = self.linear_1(x)
          return x
    model = SmallNet2()
    return model


def smallnet3(in_chan, f_lin_lay, l_lin_lay, ks, dropout=0.5):
    class SmallNet3(nn.Module):
        # 2 conv layers, 2 linear
        def __init__(self):
            super(SmallNet3, self).__init__()
            self.flatten = nn.Flatten()

            self.conv_layers = nn.Sequential(  # 1, 2, 144, 452
                  nn.Conv2d(in_channels=in_chan, out_channels=32, kernel_size=ks, padding=2),
                  nn.ReLU(), #inplace=True
                  nn.Dropout(p=dropout),
                  nn.Conv2d(in_channels =32, out_channels=64, kernel_size=ks),
                  nn.ReLU(),
                  nn.MaxPool2d(2,2),
                  nn.Dropout(p=dropout), # (1x258048 and 16384x100)
              )

            self.linear_1 = nn.Sequential(    #1x16384 and 4096x100)
                nn.Linear(f_lin_lay, 100),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(100,l_lin_lay),
                nn.Softmax(),
            )


        def forward(self, x):
          x= self.conv_layers(x)
          x = x.flatten()
          x = x.squeeze()
          x = self.linear_1(x)
          return x
    model = SmallNet3()
    return model

