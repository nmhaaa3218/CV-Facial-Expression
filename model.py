from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn
import torch.nn.functional as F
import torch

# Evaluation metric - Accuracy in this case.

import torch.nn.functional as F
Labels = {
    0:'Angry',
    1:'Disgust',
    2:'Fear',
    3:'Happy',
    4:'Sad',
    5:'Surprise',
    6:'Neutral'
}
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    # Only use non_blocking on CUDA, not MPS
    return data.to(device, non_blocking=(device.type == 'cuda'))

def accuracy(output, labels):
    predictions, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

class expression_model(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch[{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        
        
class EfficientNetB0(expression_model):
    def __init__(self, number_of_class: int, in_channel: int = 1):
        super().__init__()

        # ---- backbone (no weights) -----------------------------------------
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # ---- replace stem for 1-channel input ------------------------------
        # features[0] = ConvNormActivation(conv, bn, silu)
        old_conv = self.model.features[0][0]          # nn.Conv2d
        self.model.features[0][0] = nn.Conv2d(
            in_channels   = in_channel,
            out_channels  = old_conv.out_channels,    # 32
            kernel_size   = old_conv.kernel_size,
            stride        = old_conv.stride,
            padding       = old_conv.padding,
            bias=False
        )

        # ---- replace classifier head ---------------------------------------
        in_feat = self.model.classifier[1].in_features  # 1280 for b0
        self.model.classifier[1] = nn.Linear(in_feat, number_of_class)

    # -----------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)        # returns logits (batch, num_classes)

