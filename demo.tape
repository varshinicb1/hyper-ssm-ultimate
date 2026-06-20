# Infinite Context Memory — Demo GIF
# Requires: vhs (https://github.com/charmbracelet/vhs)
# Run: vhs demo.tape
# Output: demo.gif

Output demo.gif

Set FontSize 16
Set Width 800
Set Height 500
Set Padding 24
Set Theme "Dracula"

Type "pip install -r requirements.txt" Enter
Sleep 3s
Type "python icm_demo.py" Enter
Sleep 2s
Type "python applications/cli_chat.py --model gpt2" Enter
Sleep 3s
Type "Hello! My name is Alice." Enter
Sleep 3s
Type "What is my name?" Enter
Sleep 3s
Type "/exit" Enter
