import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Ensure module visibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hyper_ssm.model import HyperSSM, HyperSSMConfig
from hyper_ssm.audio_topology import HyperAudioEmbedder

from hyper_ssm.cuda_ops import get_lorentz_product, get_project_to_tangent, is_cuda_available

lorentz_product = get_lorentz_product()
project_to_tangent = get_project_to_tangent()

if is_cuda_available():
    print("[SYSTEM] Loaded accelerated CUDA kernels for Hyperbolic mapping (via cuda_ops).")
else:
    print("[SYSTEM] Using PyTorch fallbacks for Riemannian ops.")

class SyntheticSpeechLoader:
    """
    A local proxy loader simulating massive continuous speech spectrogram streams.
    Generates synthetic tensors matching standard Mel-Spectrogram geometry (1x80x512) 
    and targets (10 classes - e.g., Keyword Spotting).
    """
    def __init__(self, batch_size=4, mel_bins=80, time_frames=512, num_classes=10):
        self.batch_size = batch_size
        self.mel_bins = mel_bins
        self.time_frames = time_frames
        self.num_classes = num_classes
        
    def stream_batches(self, num_batches=1000):
        # We enforce a standard deviation (0.5) to mimic pre-normalized audio spectral data.
        # This will test the Riemannian exponential mappings against continuous sound-wave vectors.
        for _ in range(num_batches):
            audio = torch.randn(self.batch_size, 1, self.mel_bins, self.time_frames) * 0.5
            labels = torch.randint(0, self.num_classes, (self.batch_size,))
            yield audio, labels

class SpeechHyperClassifier(nn.Module):
    """
    Wraps the core Hyper-SSM logic securely around the Audio Embedder and a Classification Head.
    """
    def __init__(self, config, num_classes=10):
        super().__init__()
        self.embedder = HyperAudioEmbedder(
            mel_bins=80, 
            patch_time=4, 
            hidden_size=config.hidden_size,
            max_seq_len=2048 # Provide ample positional headroom for long speech files
        )
        self.core = HyperSSM(config)
        self.head = nn.Linear(config.hidden_size, num_classes)
        
    def forward(self, mel_tensor, return_entropy=False):
        # 1. Slice raw frequencies into a continuous topological temporal sequence
        # Example Output: [B, 128, Hidden_Size] (for 512 time frames with patch=4)
        hidden_states = self.embedder(mel_tensor)
        
        # 2. Inject natively into the non-Euclidean manifold layers
        # Bypass the discrete LM Head entirely since we are in Continuous Audio space
        entropy = 0.0
        for layer in self.core.layers:
            hidden_states, layer_entropy = layer(hidden_states)
            entropy += layer_entropy
            
        hidden_states = self.core.ln_f(hidden_states)
        
        # 3. Pull the geometrically folded context state from the sequence terminus
        # The Hyper-SSM folds entire speech utterances into this single static vector.
        folded_speech_descriptor = hidden_states[:, -1, :] # [B, Hidden_Size]
        
        # 4. Project the manifold state back to Euclidean keyword/speaker probabilities
        class_logits = self.head(folded_speech_descriptor)
        
        if return_entropy:
            return class_logits, entropy
        return class_logits

def main():
    print("=== HYPER-SSM STARTING SCALE-UP (Continuous Audio Topology) ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on: {device}")
    
    # Scale-Up Configuration: ~50M Parameters
    # Note: `vocab_size` is legally required by the base `HyperSSMConfig` class, 
    # but the actual embedding layer is bypassed in runtime when passing `inputs_embeds`.
    config = HyperSSMConfig(
        vocab_size=1, 
        hidden_size=256,    
        num_layers=12       
    )
    
    # Initialize the Multimodal Wrapper Model
    model = SpeechHyperClassifier(config, num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params / 1e6:.2f} M\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Enable BF16 processing natively. 
    print("Enabling strict bfloat16 mixed precision execution for geometric stability...")
    torch.set_float32_matmul_precision("medium")
    model = model.bfloat16()
    
    loader = SyntheticSpeechLoader(batch_size=8)
    
    epochs = 1
    print_interval = 20
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        
        batch_idx = 0
        for X, Y in loader.stream_batches(num_batches=200):
            X, Y = X.to(device).bfloat16(), Y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass using absolute continuous Multimodal paths
            class_logits, entropy = model(X, return_entropy=True)
            
            # Loss Function: Keyword spotting classification + Entropy penalty 
            task_loss = criterion(class_logits, Y)
            loss = task_loss - 0.01 * entropy
            
            loss.backward()
            
            # Riemannian Gradient Adjustments protect the audio patching logic mapped to `W_state`.
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if 'W_state.weight' in name or 'W_input.weight' in name:
                            param.grad = project_to_tangent(param, param.grad)
                            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()
            
            if batch_idx % print_interval == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx:05d} | Loss: {loss.item():.4f} | R-Entropy: {entropy.item():.4f}")
                
            batch_idx += 1
            
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
