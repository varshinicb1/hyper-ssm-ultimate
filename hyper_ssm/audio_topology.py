import torch
import torch.nn as nn

class HyperAudioEmbedder(nn.Module):
    """
    Transforms continuous audio Mel-Spectrograms [Batch, 1, Mel_Bins, Time_Frames] 
    into a discrete sequence of continuous vectors [Batch, seq_len, hidden_size] 
    optimized for the non-Euclidean Hyper-SSM geometry.
    
    This acts as the bridge allowing the Riemannian manifold to comprehend raw Speech
    without relying on external forced-alignment tokenizers (like Whisper).
    """
    def __init__(self, mel_bins=80, patch_time=4, hidden_size=256, max_seq_len=1024):
        super().__init__()
        self.mel_bins = mel_bins
        self.patch_time = patch_time
        self.hidden_size = hidden_size
        
        # Audio Spectrograms fundamentally differ from Vision grids. We care deeply about the 
        # continuous progression across the Time-axis `T` while compressing the entire Frequency-axis `F`.
        # We use a 2D Convolution mapping [1 x Mel_Bins x Patch_Time] -> [Hidden_Size x 1 x 1] instantly.
        self.audio_embed = nn.Conv2d(
            in_channels=1, 
            out_channels=hidden_size, 
            kernel_size=(mel_bins, patch_time), 
            stride=(1, patch_time) # Stride only across time. Frequencies are fully consumed.
        )
        
        # Absolute positional embeddings matching the temporal sequence arrival path
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        
        # We enforce a strict initial Euclidean standard deviation to prevent the initial
        # continuous audio-variance maps from instantly blowing up `cosh(x)` inside `stable_expmap`
        nn.init.trunc_normal_(self.pos_embed, std=0.01)
        nn.init.trunc_normal_(self.audio_embed.weight, std=0.01)

    def forward(self, x):
        """
        Input x: [Batch, Channels, Mel_Bins, Time_Frames] (e.g., [4, 1, 80, 512])
        Returns: [Batch, Sequence_Len, Hidden_Size] (e.g., [4, 128, 256])
        """
        B, C, F, T = x.shape
        assert F == self.mel_bins, \
            f"Input spectrogram frequency {F} does not match initialized {self.mel_bins} Mel-Bins."
            
        # 1. Project the continuous temporal spectrogram into the target depth
        # Output shape: [B, hidden_size, 1, T/patch_time]
        x = self.audio_embed(x)
        
        # 2. Flatten spatial properties into a 1D sequence
        # Shape: [B, hidden_size, 1, seq_len] -> [B, hidden_size, seq_len] -> Transpose -> [B, seq_len, hidden_size]
        x = x.squeeze(2).transpose(1, 2)
        
        seq_len = x.shape[1]
        
        # 3. Apply positional topology, slicing to the exact temporal duration
        x = x + self.pos_embed[:, :seq_len, :]
        
        return x
