import torch
import torch.nn as nn

class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) for word-level recognition.
    CNN extracts features -> RNN models sequence -> CTC Loss (used during training).
    """
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        # CNN layers (Feature extraction)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)) # Only pool vertically for sequences
        )
        
        # RNN layers (Sequence modeling)
        self.rnn = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        
        # Fully connected (Mapping to classes)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        # x shape: (N, 1, H, W)
        conv_out = self.cnn(x) # (N, C, H', W')
        
        # Reshape for RNN: (N, W', C * H')
        b, c, h, w = conv_out.size()
        conv_out = conv_out.view(b, c * h, w)
        conv_out = conv_out.permute(0, 2, 1) # (N, W', Hidden)
        
        rnn_out, _ = self.rnn(conv_out)
        
        output = self.fc(rnn_out)
        return output # (Batch, SequenceLength, NumClasses)

# Note: Word recognition requires a dataset of word images (like IAM Dataset) 
# and training with Connectionist Temporal Classification (CTC) loss.
