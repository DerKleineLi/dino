"""SegmentationNN"""
import torch
import torch.nn as nn
from einops import rearrange


class SegmentationNN(nn.Module):
    def __init__(self, backbone, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.backbone = backbone
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.Conv2d(384, 192, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.Conv2d(192, 96, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(96, num_classes, 1, 1, 0),
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.backbone(x, segm=True)  # (B P 384)
        B, P, D = x.shape
        x = x.view(B, 14, 14, D)
        x = rearrange(x, "b w h d -> b d w h")
        x = self.decoder(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print("Saving model... %s" % path)
        torch.save(self, path)


class DummySegmentationModel(nn.Module):
    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()


if __name__ == "__main__":
    from torchinfo import summary

    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")
