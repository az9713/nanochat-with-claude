# nanochat Quick Start Guide

Welcome to nanochat! This guide will help you get started quickly with hands-on examples. Each use case is designed to give you a quick win and build your confidence before moving to more advanced features.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Use Case 1: Chat with a Pre-trained Model](#use-case-1-chat-with-a-pre-trained-model)
4. [Use Case 2: Ask Questions via Command Line](#use-case-2-ask-questions-via-command-line)
5. [Use Case 3: Use the ChatGPT-style Web Interface](#use-case-3-use-the-chatgpt-style-web-interface)
6. [Use Case 4: Adjust Response Creativity with Temperature](#use-case-4-adjust-response-creativity-with-temperature)
7. [Use Case 5: Train a Tiny Model on Your Computer](#use-case-5-train-a-tiny-model-on-your-computer)
8. [Use Case 6: Evaluate Your Model's Knowledge](#use-case-6-evaluate-your-models-knowledge)
9. [Use Case 7: Test Math Problem Solving](#use-case-7-test-math-problem-solving)
10. [Use Case 8: Generate Creative Writing](#use-case-8-generate-creative-writing)
11. [Use Case 9: Train a Custom Tokenizer](#use-case-9-train-a-custom-tokenizer)
12. [Use Case 10: Run the Full GPT-2 Training Pipeline](#use-case-10-run-the-full-gpt-2-training-pipeline)
13. [Use Case 11: Monitor Training with Weights & Biases](#use-case-11-monitor-training-with-weights--biases)
14. [Use Case 12: Customize Your AI's Personality](#use-case-12-customize-your-ais-personality)
15. [Next Steps](#next-steps)

---

## Prerequisites

Before you begin, make sure you have:

1. **A computer running Linux, macOS, or Windows** (with WSL for Windows users)
2. **Python 3.10 or higher** installed
3. **At least 8GB of RAM** (16GB+ recommended)
4. **At least 10GB of free disk space**
5. **A GPU is optional** but highly recommended for training

### How to Check Python Version

Open your terminal (Command Prompt on Windows, Terminal on macOS/Linux) and type:

```bash
python3 --version
```

You should see something like `Python 3.10.x` or higher. If not, install Python from [python.org](https://www.python.org/downloads/).

---

## Installation

### Step 1: Download nanochat

Open your terminal and run these commands one by one:

```bash
# Navigate to where you want to store nanochat
cd ~

# Download nanochat (if using git)
git clone https://github.com/karpathy/nanochat.git

# Enter the nanochat directory
cd nanochat
```

**Don't have git?** You can download the ZIP file from GitHub and extract it instead.

### Step 2: Install the uv Package Manager

nanochat uses `uv`, a fast Python package manager. Install it:

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

After installation, restart your terminal.

### Step 3: Create Virtual Environment and Install Dependencies

```bash
# For computers with NVIDIA GPU:
uv sync --extra gpu

# For computers without GPU (CPU only):
uv sync --extra cpu
```

**What this does:** Creates an isolated Python environment with all required packages. This prevents conflicts with other Python projects on your computer.

### Step 4: Activate the Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```cmd
.venv\Scripts\activate
```

You should see `(.venv)` appear at the beginning of your terminal prompt. This means the virtual environment is active.

**Important:** You need to activate the virtual environment every time you open a new terminal to work with nanochat.

---

## Use Case 1: Chat with a Pre-trained Model

**Goal:** Have an interactive conversation with an AI you trained (or a pre-trained model).

**Prerequisites:** You need a trained model. If you haven't trained one yet, skip to [Use Case 5](#use-case-5-train-a-tiny-model-on-your-computer) first.

### Steps:

1. Make sure your virtual environment is activated (you see `(.venv)` in your prompt)

2. Start the interactive chat:
   ```bash
   python -m scripts.chat_cli
   ```

3. You'll see a prompt like:
   ```
   NanoChat Interactive Mode
   --------------------------------------------------
   Type 'quit' or 'exit' to end the conversation
   Type 'clear' to start a new conversation
   --------------------------------------------------

   User:
   ```

4. Type your message and press Enter. The AI will respond!

5. Type `quit` to exit when done.

### Example Conversation:
```
User: Hello! How are you?
Assistant: Hello! I'm doing well, thank you for asking. How can I help you today?

User: What is the capital of France?
Assistant: The capital of France is Paris.

User: quit
Goodbye!
```

---

## Use Case 2: Ask Questions via Command Line

**Goal:** Get a quick answer without entering interactive mode.

### Steps:

1. Use the `-p` flag to pass your question directly:
   ```bash
   python -m scripts.chat_cli -p "What is the speed of light?"
   ```

2. The AI will print its response and exit automatically.

### More Examples:
```bash
# Ask about science
python -m scripts.chat_cli -p "Why is the sky blue?"

# Ask for a definition
python -m scripts.chat_cli -p "What is machine learning?"

# Ask for a list
python -m scripts.chat_cli -p "Name three planets in our solar system"
```

**Tip:** This is useful for scripting or when you just need a quick answer!

---

## Use Case 3: Use the ChatGPT-style Web Interface

**Goal:** Chat with your AI through a beautiful web interface, just like ChatGPT.

### Steps:

1. Start the web server:
   ```bash
   python -m scripts.chat_web
   ```

2. Wait for the server to start. You'll see:
   ```
   Starting NanoChat Web Server
   Temperature: 0.8, Top-k: 50, Max tokens: 512
   Loading nanochat models across GPUs...
   All 1 workers initialized!
   Server ready at http://localhost:8000
   ```

3. Open your web browser and go to: `http://localhost:8000`

4. You'll see a ChatGPT-like interface. Type your message in the text box and press Enter!

5. To stop the server, go back to your terminal and press `Ctrl+C`.

### Customizing the Web Server:

```bash
# Use a different port (if 8000 is busy)
python -m scripts.chat_web --port 3000

# Adjust default temperature (creativity)
python -m scripts.chat_web --temperature 0.5

# Use multiple GPUs (if available)
python -m scripts.chat_web --num-gpus 4
```

---

## Use Case 4: Adjust Response Creativity with Temperature

**Goal:** Learn how temperature affects AI responses.

### Understanding Temperature:

- **Temperature = 0.0**: Deterministic, always gives the same response
- **Temperature = 0.6**: Slightly creative, good for factual Q&A
- **Temperature = 1.0**: Balanced creativity
- **Temperature = 1.5+**: Very creative, may produce unexpected results

### Steps:

1. Try low temperature (factual):
   ```bash
   python -m scripts.chat_cli -p "What is 2+2?" -t 0.0
   ```

2. Try higher temperature (creative):
   ```bash
   python -m scripts.chat_cli -p "Write a poem about the moon" -t 1.0
   ```

3. Compare the same prompt with different temperatures:
   ```bash
   # Deterministic
   python -m scripts.chat_cli -p "Tell me a story" -t 0.0

   # Creative
   python -m scripts.chat_cli -p "Tell me a story" -t 1.2
   ```

**Observation:** Higher temperatures produce more varied and creative responses, but may also be less coherent.

---

## Use Case 5: Train a Tiny Model on Your Computer

**Goal:** Train your very first language model, even without a GPU!

### What You'll Learn:
- How the training process works
- What the output means
- How to verify training completed

### Steps:

1. First, train a tokenizer (this is quick):
   ```bash
   python -m scripts.tok_train
   ```

   **What this does:** Creates a vocabulary of "tokens" that the model will use to understand text.

2. Train a very small model (CPU-friendly):
   ```bash
   python -m scripts.base_train \
       --depth=4 \
       --max-seq-len=512 \
       --device-batch-size=1 \
       --total-batch-size=512 \
       --num-iterations=20 \
       --core-metric-every=-1
   ```

   **What the parameters mean:**
   - `--depth=4`: A very small model (4 layers instead of 20+)
   - `--max-seq-len=512`: Maximum text length to process
   - `--device-batch-size=1`: Process one example at a time (less memory)
   - `--num-iterations=20`: Only 20 training steps (quick demo)

3. Watch the output:
   ```
   step 00001/00020 (5.00%) | loss: 10.234567 | dt: 1234ms | tok/sec: 415
   step 00002/00020 (10.00%) | loss: 9.876543 | dt: 1200ms | tok/sec: 426
   ...
   ```

   **What to look for:** The `loss` should decrease over time. This means the model is learning!

4. After training, a checkpoint is saved in `~/.cache/nanochat/base_checkpoints/d4/`

**Congratulations!** You just trained your first language model!

---

## Use Case 6: Evaluate Your Model's Knowledge

**Goal:** Test how well your model performs on standardized benchmarks.

### Steps:

1. After training, evaluate on the CORE benchmark:
   ```bash
   python -m scripts.base_eval --eval core
   ```

2. The output shows accuracy on various tasks:
   ```
   Evaluating: hellaswag (10-shot, type: multiple_choice)... accuracy: 0.2543
   Evaluating: arc_easy (0-shot, type: multiple_choice)... accuracy: 0.3125
   ...
   CORE metric: 0.1234
   ```

3. Evaluate the chat model on question-answering:
   ```bash
   python -m scripts.chat_eval -i sft -a ARC-Easy
   ```

### Understanding the CORE Metric:
- **0.0**: Random guessing (no knowledge)
- **0.256**: GPT-2 level (our target!)
- **Higher**: Better than GPT-2

**Note:** Small models trained briefly won't score high, but you can see the evaluation process works!

---

## Use Case 7: Test Math Problem Solving

**Goal:** See how your model handles math problems with the built-in calculator tool.

### How It Works:
nanochat models learn to use a calculator tool. When they need to compute something, they write the expression in a special format, and the system evaluates it.

### Steps:

1. Start the chat and ask a math question:
   ```bash
   python -m scripts.chat_cli
   ```

2. Try these prompts:
   ```
   User: What is 15 times 37?
   User: If I have 100 dollars and spend 35%, how much do I have left?
   User: Calculate 2.5 * 4.8 + 3.2
   ```

3. The model will (attempt to) use its calculator tool to compute the answer.

**Note:** Math abilities depend heavily on training. The full SFT training includes GSM8K math problems.

---

## Use Case 8: Generate Creative Writing

**Goal:** Use the model for creative tasks like storytelling.

### Steps:

1. Start the web interface for a better experience:
   ```bash
   python -m scripts.chat_web --temperature 1.0
   ```

2. Open `http://localhost:8000` in your browser

3. Try these creative prompts:
   - "Write a short story about a robot who learns to paint"
   - "Compose a haiku about programming"
   - "Describe a sunset as if you were a pirate"
   - "Create a recipe for an imaginary dish called 'Cloud Soup'"

4. Experiment with different temperature settings to vary creativity

### Tips for Better Results:
- Be specific in your prompts
- Give context and constraints
- Higher temperature = more surprising results
- Lower temperature = more coherent but predictable

---

## Use Case 9: Train a Custom Tokenizer

**Goal:** Understand how text is converted to numbers for the model.

### What is a Tokenizer?
A tokenizer breaks text into "tokens" - pieces that the model understands. For example:
- "Hello world" might become tokens: `["Hello", " world"]`
- These become numbers: `[15496, 995]`

### Steps:

1. First, download training data:
   ```bash
   python -m nanochat.dataset -n 2
   ```

   **What this does:** Downloads 2 data shards (~200MB) from FineWeb-Edu.

2. Train the tokenizer:
   ```bash
   python -m scripts.tok_train
   ```

   This creates a vocabulary of 32,768 tokens.

3. Evaluate the tokenizer:
   ```bash
   python -m scripts.tok_eval
   ```

   Output shows the compression ratio (how efficiently text is encoded).

### Where is the Tokenizer Saved?
The tokenizer is saved to `~/.cache/nanochat/tokenizer/`

---

## Use Case 10: Run the Full GPT-2 Training Pipeline

**Goal:** Train a complete GPT-2 grade model from scratch.

### Requirements:
- **8×H100 GPUs** (or equivalent high-end setup)
- **~3 hours of compute time**
- **~$73 in cloud compute costs**

### Steps:

1. **Rent a GPU server** from a cloud provider:
   - [Lambda Labs](https://lambda.ai/service/gpu-cloud)
   - [RunPod](https://www.runpod.io/)
   - [Vast.ai](https://vast.ai/)

   Choose an instance with 8×H100 or 8×A100 GPUs.

2. **SSH into your server** and clone nanochat:
   ```bash
   git clone https://github.com/karpathy/nanochat.git
   cd nanochat
   ```

3. **Run the full pipeline:**
   ```bash
   bash runs/speedrun.sh
   ```

4. **What happens:**
   - Environment setup (~2 min)
   - Tokenizer training (~5 min)
   - Data download (~10 min)
   - Pretraining (~3 hours)
   - SFT training (~30 min)
   - Evaluation (~10 min)

5. **After training, chat with your model:**
   ```bash
   python -m scripts.chat_web
   ```

### Monitoring Progress:
```
step 00100/16704 (0.60%) | loss: 8.123456 | mfu: 45.2 | eta: 180.5m
step 00200/16704 (1.20%) | loss: 7.654321 | mfu: 46.1 | eta: 175.3m
```

- **loss**: Should decrease (model is learning)
- **mfu**: Model FLOPs Utilization (higher is better, 40-50% is good)
- **eta**: Estimated time remaining

---

## Use Case 11: Monitor Training with Weights & Biases

**Goal:** Visualize training progress with beautiful charts.

### What is Weights & Biases (W&B)?
A free tool that creates real-time charts of your training metrics.

### Steps:

1. **Create a free W&B account** at [wandb.ai](https://wandb.ai)

2. **Log in from terminal:**
   ```bash
   wandb login
   ```

   Paste your API key when prompted.

3. **Run training with W&B enabled:**
   ```bash
   python -m scripts.base_train --run=my-experiment
   ```

   The `--run` flag sets the experiment name (anything except "dummy" enables W&B).

4. **View your dashboard:**
   - Go to [wandb.ai](https://wandb.ai)
   - Find your project "nanochat"
   - Click on your run to see charts!

### What You'll See:
- Training loss over time
- Validation metrics
- Learning rate schedule
- GPU utilization

---

## Use Case 12: Customize Your AI's Personality

**Goal:** Give your nanochat a unique personality through custom training data.

### How It Works:
During SFT (Supervised Fine-Tuning), the model learns from conversation examples. You can add your own conversations to shape its personality!

### Steps:

1. **Create a custom conversations file:**

   Create a file called `my_identity.jsonl` with this format:
   ```json
   {"messages": [{"role": "user", "content": "What is your name?"}, {"role": "assistant", "content": "I'm NanoBot, your friendly AI assistant!"}]}
   {"messages": [{"role": "user", "content": "Who created you?"}, {"role": "assistant", "content": "I was created using the nanochat framework."}]}
   {"messages": [{"role": "user", "content": "What do you like?"}, {"role": "assistant", "content": "I enjoy helping people learn about AI and technology!"}]}
   ```

   Each line is a complete conversation in JSON format.

2. **Place the file in the nanochat cache:**
   ```bash
   cp my_identity.jsonl ~/.cache/nanochat/
   ```

3. **Modify `scripts/chat_sft.py`** to include your file:
   ```python
   # Add to the imports
   from tasks.customjson import CustomJSON

   # Add to train_dataset TaskMixture
   CustomJSON(filepath="/path/to/my_identity.jsonl"),
   ```

4. **Run SFT training:**
   ```bash
   python -m scripts.chat_sft
   ```

### Tips for Good Training Data:
- Write diverse examples (different topics, tones)
- Include at least 100-1000 examples
- Make conversations natural and helpful
- Be consistent with the personality you want

---

## Next Steps

Congratulations on completing the Quick Start Guide! Here's what to explore next:

### For Users:
1. Read the [User Guide](USER_GUIDE.md) for comprehensive usage instructions
2. Experiment with different prompts and temperatures
3. Try the web interface with friends

### For Developers:
1. Read the [Developer Guide](DEVELOPER_GUIDE.md) to understand the codebase
2. Study the [Architecture Guide](ARCHITECTURE.md) for technical deep-dives
3. Look at `nanochat/gpt.py` to understand the model architecture
4. Explore `tasks/` to see how evaluations work

### Advanced Topics:
1. **Reinforcement Learning**: See `scripts/chat_rl.py`
2. **Custom Evaluation Tasks**: Create new tasks in `tasks/`
3. **Model Architecture Changes**: Modify `nanochat/gpt.py`
4. **Optimization Research**: Experiment with `nanochat/optim.py`

### Getting Help:
- [DeepWiki](https://deepwiki.com/karpathy/nanochat): Ask questions about the repo
- [GitHub Discussions](https://github.com/karpathy/nanochat/discussions): Community help
- [Discord #nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031): Real-time chat

---

## Troubleshooting Common Issues

### "ModuleNotFoundError: No module named 'nanochat'"
**Solution:** Make sure you're in the nanochat directory and your virtual environment is activated.
```bash
cd ~/nanochat
source .venv/bin/activate
```

### "CUDA out of memory"
**Solution:** Reduce batch size:
```bash
python -m scripts.base_train --device-batch-size=1
```

### "Tokenizer not found"
**Solution:** Train the tokenizer first:
```bash
python -m scripts.tok_train
```

### "Model checkpoint not found"
**Solution:** Train a base model first:
```bash
python -m scripts.base_train --depth=4 --num-iterations=20
```

### Web interface shows blank page
**Solution:** Check the console for errors. Make sure port 8000 isn't blocked by a firewall.

---

Happy learning with nanochat! 🎉
