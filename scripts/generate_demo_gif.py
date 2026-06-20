"""Generate a demo GIF showing ICM O(1) memory output."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.font_manager import findfont, FontProperties
import os

# Try to find a monospace font
try:
    mono_font = findfont(FontProperties(family=["Consolas", "DejaVu Sans Mono", "Courier New", "monospace"]))
except Exception:
    mono_font = None

W, H = 800, 480
DPI = 96

lines = [
    "  =======================================================",
    "  Infinite Context Memory (ICM) -- O(1) Hyperbolic Memory",
    "  =======================================================",
    "",
    "   #  Speaker   Message                              Memory",
    "  --  --------  ---------------------------------  --------",
    "   1  Alice     Hello, my name is Alice.                260B",
    "   2  Alice     I am a software engineer from San ...   260B",
    "   3  Alice     I work at a startup building AI to...   260B",
    "   4  Alice     My favorite programming language i...   260B",
    "   5  Alice     I also enjoy hiking and photography.    260B",
    "   6  Alice     I have a golden retriever named Max.    260B",
    "   7  Alice     Max is 3 years old and loves to fetch.  260B",
    "   8  Alice     I live in the Mission District.         260B",
    "   9  Alice     My favorite food is ramen.              260B",
    "  10  Bot       What is my dog name and where do ...   260B",
    "",
    "  =======================================================",
    "  [OK] Final memory: 260 bytes  (fixed, O(1))",
    "  [OK] Turns stored: 10",
    "  [OK] State dim: 64,  Scales: 4",
    "  =======================================================",
]

N_FRAMES = len(lines)
FPS = 4

def make_frame(i):
    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, W/DPI)
    ax.set_ylim(0, H/DPI)
    ax.axis("off")

    y = H/DPI - 30/DPI
    for j, line in enumerate(lines[:i+1]):
        color = "#c9d1d9"
        if "OK" in line:
            color = "#3fb950"
        if "ICM" in line or "Infinite" in line:
            color = "#58a6ff"
        if "===" in line:
            color = "#30363d"
        if "Memory" in line and "bytes" in line:
            color = "#d29922"
        ax.text(
            12/DPI, y, line,
            fontfamily="Consolas" if mono_font else "monospace",
            fontsize=10, color=color, va="top",
        )
        y -= 22/DPI

    plt.tight_layout(pad=0)
    return fig

frames = []
for i in range(N_FRAMES):
    fig = make_frame(i)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frames.append(buf)
    plt.close(fig)

import imageio
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.gif")
imageio.mimsave(out_path, frames, fps=FPS, loop=0)
print(f"Saved {out_path}  ({len(frames)} frames)")
