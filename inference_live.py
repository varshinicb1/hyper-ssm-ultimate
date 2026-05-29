import torch
import cv2
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import sys
import os

# Ensure module visibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hyper_ssm.model import HyperSSMConfig
from train_vision import VisionHyperClassifier
from train_audio import SpeechHyperClassifier
import torchaudio.transforms as T

# ======================================================================================
# LIVE MULTIMODAL INFERENCE DEPLOYMENT
# ======================================================================================

class MultimodalHyperSSMDeployment:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        print(f"Initializing Native Tri-Modal Inference Engine on [{self.device}]...")
        
        # Load unified Tri-Modal Configuration (e.g., 50M parameter geometry limit)
        self.config = HyperSSMConfig(vocab_size=1, hidden_size=256, num_layers=12)
        
        # 1. Initialize Vision Manifold
        self.vision_model = VisionHyperClassifier(self.config, num_classes=1000).to(self.device).bfloat16()
        self.vision_model.eval()
        
        # 2. Initialize Audio Manifold
        self.audio_model = SpeechHyperClassifier(self.config, num_classes=10).to(self.device).bfloat16()
        self.audio_model.eval()
        
        # Real-time Stream Queues
        self.vision_queue = queue.Queue(maxsize=2)
        self.audio_queue = queue.Queue(maxsize=5)
        
        # Audio Configuration Parameters
        self.sample_rate = 16000
        self.audio_chunk_size = 8000 # 500ms sliding overlapping window
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80
        ).to(self.device)
        
        self.running = True

    # ----------------------------------------------------------------------------------
    # DATA INGESTION: VISION (OPENCV)
    # ----------------------------------------------------------------------------------
    def capture_vision(self):
        cap = cv2.VideoCapture(0)
        print("[VISION] Camera Stream Active.")
        
        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            
            # Map OpenCV BGR to standard RGB [3, 224, 224] continuous Tensor geometry
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            
            # Normalize to simulated ImageNet distributions [-1, 1] variance
            tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float()
            tensor = (tensor / 127.5) - 1.0 
            tensor = tensor.unsqueeze(0).to(self.device).bfloat16()
            
            if not self.vision_queue.full():
                self.vision_queue.put(tensor)
                
            # Render visual box for user preview
            cv2.imshow("Hyper-SSM Live Visual Sensor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                
        cap.release()
        cv2.destroyAllWindows()

    # ----------------------------------------------------------------------------------
    # DATA INGESTION: AUDIO (SOUNDDEVICE)
    # ----------------------------------------------------------------------------------
    def audio_callback(self, indata, frames, time, status):
        """ Sliding window raw capture """
        if status: print(status)
        if not self.audio_queue.full():
            # Raw wave to Tensor [1, 1, Samples]
            wave = torch.from_numpy(indata.copy()).squeeze(-1).float()
            wave = wave.unsqueeze(0).to(self.device)
            self.audio_queue.put(wave)

    def capture_audio(self):
        print("[AUDIO] Microphone Stream Active.")
        with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                            blocksize=self.audio_chunk_size, callback=self.audio_callback):
            while self.running:
                time.sleep(0.1)

    # ----------------------------------------------------------------------------------
    # INFERENCE ENGINE LOOP
    # ----------------------------------------------------------------------------------
    def run_inference(self):
        print("\n=== HYPER-SSM LIVE MULTIMODAL INFERENCE PIPELINE ===")
        print("Waiting for streaming buffers to initialize...\n")
        time.sleep(2)
        
        while self.running:
            vision_tensor = None
            audio_wave = None
            
            # Dequeue latest available sensory topological streams safely
            if not self.vision_queue.empty(): 
                vision_tensor = self.vision_queue.get()
            if not self.audio_queue.empty(): 
                audio_wave = self.audio_queue.get()

            with torch.no_grad():
                # --- VISUAL CONTINUOUS CLASSIFICATION ---
                if vision_tensor is not None:
                    vision_logits, v_entropy = self.vision_model(vision_tensor, return_entropy=True)
                    v_confidence = torch.softmax(vision_logits, dim=-1).max().item()
                    v_class = torch.argmax(vision_logits, dim=-1).item()
                    
                    print(f"[VISION] Class {v_class:03d} | Confidence: {v_confidence*100:04.1f}% | R-Entropy Limit: {v_entropy.item():.2f}")

                # --- AUDIO TEMPORAL CLASSIFICATION ---
                if audio_wave is not None:
                    # Construct Mel-Spectrogram online [Batch, 1, Mel_Bins, Time_Frames]
                    mel_spec = self.mel_transform(audio_wave)
                    mel_spec = mel_spec.unsqueeze(1).bfloat16()
                    
                    audio_logits, a_entropy = self.audio_model(mel_spec, return_entropy=True)
                    a_confidence = torch.softmax(audio_logits, dim=-1).max().item()
                    a_class = torch.argmax(audio_logits, dim=-1).item()
                    
                    print(f"[AUDIO]  Keyword {a_class:02d}  | Confidence: {a_confidence*100:04.1f}% | R-Entropy Limit: {a_entropy.item():.2f}")

            time.sleep(0.01) # Allow IO threads to breathe

def main():
    # Only run native cuda bindings if hardware matches compiler flag
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deploy = MultimodalHyperSSMDeployment(device=device)
    
    # Launch parallel sensory intake threads
    t_vision = threading.Thread(target=deploy.capture_vision)
    t_audio = threading.Thread(target=deploy.capture_audio)
    
    t_vision.start()
    t_audio.start()
    
    # Launch Inference Engine on Main Thread
    try:
        deploy.run_inference()
    except KeyboardInterrupt:
        deploy.running = False
        print("\nShutting down multimodal deployment pipelines...")
        
    t_vision.join()
    t_audio.join()

if __name__ == "__main__":
    main()
