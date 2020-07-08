import torch
from torch import nn
from torchvision import models


class Disentangler(nn.Module):
    def __init__(self):
        super(Disentangler, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True, progress=False, num_classes=1000)
        self.fc1 = nn.Linear(64, 48)
        self.fc2 = nn.Linear(64, 48)
        self.relu = nn.ReLU()

        self.conv_idx_list = [3, 8, 13, 22, 31]  # idxes of conv layers in resnet50 cf.paper

    def extract_features(self, img):
        # define hook
        def vgg_x_hook(module, input, output):
            output.detach_()  # no gradient compute
            resnet_features.append(output)

        resnet_features = []
        vgg_x_handles = []

        conv_idx_iter = 0

        # place hooks
        for i, m in enumerate(self.VGG19.features.modules()):
            if i == self.conv_idx_list[conv_idx_iter]:
                if conv_idx_iter < len(self.conv_idx_list) - 1:
                    conv_idx_iter += 1
                vgg_x_handles.append(m.register_forward_hook(vgg_x_hook))

        # run model for x
        with torch.no_grad():
            self.resnet50(img)

        # retrieve features for x
        for h in vgg_x_handles:
            h.remove()

        return resnet_features

    def forward(self, img, mean_landmark):
        features = self.extract_features(img)

        # concat with mean landmark ??

        # out = self.fc1()
        # out = self.fc2(out)
        # out = self.relu(out)
        # return out
