import torch
import torch.nn as nn

class HyperVisionEmbedder(nn.Module):
    """
    Transforms continuous visual pixel data [B, C, H, W] into a discrete sequence 
    of continuous vectors [B, seq_len, hidden_size] optimized for the Hyper-SSM geometry.
    
    This acts as the bridge allowing non-Euclidean manifolds to comprehend ImageNet or video frames
    without standard ConvNet reduction loops.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, hidden_size=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        # Calculate resulting sequence length mathematically
        assert img_size % patch_size == 0, "Image size must be strictly divisible by sequence patch size."
        self.num_patches = (img_size // patch_size) ** 2
        
        # We use a pure linear Conv2D projection to fold a [16x16x3] grid region 
        # instantly into a single flat coordinate mapped exactly to the Hyper-SSM hidden space.
        # Stride = patch_size ensures patches are mathematically orthogonal and non-overlapping.
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=hidden_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Absolute Positional Encodings since Hyperbolic operations fold sequences 
        # asynchronously based on arrival geometry, standard visual orientation is required
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
        # We enforce a strict initial Euclidean standard deviation to prevent the initial
        # continuous pixel-variance maps from instantly blowing up `cosh(x)` inside `stable_expmap`
        nn.init.trunc_normal_(self.pos_embed, std=0.01)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.01)

    def forward(self, x):
        """
        Input x: [Batch, Channels, Height, Width] (e.g., [4, 3, 224, 224])
        Returns: [Batch, Sequence_Len, Hidden_Size] (e.g., [4, 196, 256])
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input geometry {H}x{W} does not match initial {self.img_size}x{self.img_size} definition."
            
        # 1. Project the continuous grid geometry into the target depth
        # Output shape: [B, hidden_size, H/patch_size, W/patch_size]
        x = self.patch_embed(x)
        
        # 2. Flatten spatial properties into a 1D sequence
        # Shape: [B, hidden_size, num_patches] -> Transpose -> [B, num_patches, hidden_size]
        x = x.flatten(2).transpose(1, 2)
        
        # 3. Apply positional topology
        x = x + self.pos_embed
        
        return x
