# Remote GPU Development Guide

This guide explains how to set up SSH tunneling between a remote GPU provider (RunPod, Lambda Labs, Vast.ai, etc.) and your local development environment (VSCode, terminal). This allows you to develop locally while running nanochat on powerful cloud GPUs.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Generate SSH Keys](#step-1-generate-ssh-keys)
4. [Step 2: Set Up Your GPU Provider](#step-2-set-up-your-gpu-provider)
5. [Step 3: Connect via Terminal](#step-3-connect-via-terminal)
6. [Step 4: Set Up VSCode Remote SSH](#step-4-set-up-vscode-remote-ssh)
7. [Step 5: Port Forwarding for Web UI](#step-5-port-forwarding-for-web-ui)
8. [Provider-Specific Instructions](#provider-specific-instructions)
9. [Troubleshooting](#troubleshooting)
10. [Tips for Efficient Remote Development](#tips-for-efficient-remote-development)

---

## Overview

### What is SSH Tunneling?

SSH (Secure Shell) creates an encrypted connection between your local computer and a remote server. With SSH tunneling, you can:

- **Edit code locally** in VSCode while it runs on a remote GPU
- **Access web interfaces** (like nanochat's web UI) through your local browser
- **Transfer files** between your computer and the remote server
- **Run commands** on the remote server from your local terminal

### Architecture

```
┌─────────────────────┐         SSH Tunnel         ┌─────────────────────┐
│   Your Computer     │ ◄─────────────────────────► │   GPU Server        │
│                     │                             │   (RunPod/Lambda)   │
│  ┌───────────────┐  │                             │  ┌───────────────┐  │
│  │    VSCode     │  │      Encrypted Connection   │  │   nanochat    │  │
│  │  (local UI)   │──┼─────────────────────────────┼──│  (remote code)│  │
│  └───────────────┘  │                             │  └───────────────┘  │
│                     │                             │                     │
│  ┌───────────────┐  │      Port 8000 Forward      │  ┌───────────────┐  │
│  │   Browser     │──┼─────────────────────────────┼──│   Web Server  │  │
│  │ localhost:8000│  │                             │  │   :8000       │  │
│  └───────────────┘  │                             │  └───────────────┘  │
└─────────────────────┘                             └─────────────────────┘
```

---

## Prerequisites

Before you begin, ensure you have:

- **A computer** running Windows, macOS, or Linux
- **VSCode** installed ([download here](https://code.visualstudio.com/))
- **An account** with a GPU provider (RunPod, Lambda Labs, Vast.ai, etc.)
- **A credit card** for GPU rental (most providers charge by the hour)

### Cost Estimates

| Provider | GPU | Approximate Cost |
|----------|-----|------------------|
| RunPod | 8×H100 | ~$24-32/hour |
| Lambda Labs | 8×H100 | ~$24/hour |
| Vast.ai | 8×A100 | ~$8-15/hour |
| RunPod | 1×A100 | ~$1.5-2/hour |
| Vast.ai | 1×RTX 4090 | ~$0.40-0.80/hour |

---

## Step 1: Generate SSH Keys

SSH keys allow you to connect to remote servers without typing a password every time.

### On Windows

1. **Open PowerShell** (search for "PowerShell" in Start menu)

2. **Generate an SSH key pair:**
   ```powershell
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

3. **Press Enter** to accept the default location (`C:\Users\YourName\.ssh\id_ed25519`)

4. **Enter a passphrase** (optional but recommended) or press Enter twice for no passphrase

5. **View your public key:**
   ```powershell
   Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
   ```

6. **Copy the entire output** (it starts with `ssh-ed25519`)

### On macOS/Linux

1. **Open Terminal**

2. **Generate an SSH key pair:**
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

3. **Press Enter** to accept the default location (`~/.ssh/id_ed25519`)

4. **Enter a passphrase** (optional but recommended) or press Enter twice for no passphrase

5. **View your public key:**
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

6. **Copy the entire output** (it starts with `ssh-ed25519`)

### What You Should Have

After this step, you have two files:
- **Private key**: `~/.ssh/id_ed25519` (NEVER share this!)
- **Public key**: `~/.ssh/id_ed25519.pub` (this is what you give to servers)

---

## Step 2: Set Up Your GPU Provider

### RunPod

1. **Create an account** at [runpod.io](https://www.runpod.io/)

2. **Add your SSH public key:**
   - Go to **Settings** → **SSH Public Keys**
   - Click **Add SSH Key**
   - Paste your public key (from Step 1)
   - Give it a name (e.g., "My Laptop")
   - Click **Save**

3. **Deploy a GPU Pod:**
   - Go to **Pods** → **Deploy**
   - Choose a template: Select **RunPod Pytorch** or **Ubuntu**
   - Select GPU type: **8x H100 80GB** for full training, or smaller for testing
   - Select volume size: **100GB+** for nanochat
   - Click **Deploy**

4. **Wait for the pod to start** (usually 1-5 minutes)

5. **Get connection info:**
   - Click on your pod
   - Look for **SSH over exposed TCP** section
   - Note the **hostname** and **port** (e.g., `ssh root@194.68.xxx.xxx -p 12345`)

### Lambda Labs

1. **Create an account** at [lambda.ai](https://lambda.ai/)

2. **Add your SSH public key:**
   - Go to **SSH Keys** in dashboard
   - Click **Add SSH Key**
   - Paste your public key
   - Click **Add**

3. **Launch an instance:**
   - Go to **Instances** → **Launch Instance**
   - Select **8x H100** (or your preferred GPU)
   - Select your SSH key
   - Click **Launch**

4. **Get connection info:**
   - Once running, copy the **IP address** shown
   - Connection: `ssh ubuntu@<IP_ADDRESS>`

### Vast.ai

1. **Create an account** at [vast.ai](https://vast.ai/)

2. **Add your SSH public key:**
   - Go to **Account** → **SSH Keys**
   - Click **Add SSH Key**
   - Paste your public key

3. **Rent a GPU:**
   - Go to **Search** (or **Create**)
   - Filter by GPU type, price, etc.
   - Click **Rent** on a suitable offer
   - Choose **SSH** as connection type

4. **Get connection info:**
   - Go to **Instances**
   - Click on your instance
   - Copy the SSH command shown

---

## Step 3: Connect via Terminal

### Basic SSH Connection

**On Windows (PowerShell or Windows Terminal):**
```powershell
# For RunPod (note the -p flag for custom port)
ssh root@194.68.xxx.xxx -p 12345

# For Lambda Labs (default port 22)
ssh ubuntu@194.68.xxx.xxx

# For Vast.ai (varies, check your instance)
ssh -p 12345 root@ssh.vast.ai
```

**On macOS/Linux:**
```bash
# For RunPod
ssh root@194.68.xxx.xxx -p 12345

# For Lambda Labs
ssh ubuntu@194.68.xxx.xxx

# For Vast.ai
ssh -p 12345 root@ssh.vast.ai
```

### First Connection

When connecting for the first time, you'll see:
```
The authenticity of host '[194.68.xxx.xxx]:12345' can't be established.
ED25519 key fingerprint is SHA256:xxxxxxxxxxxxxxxxxxxx.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Type `yes` and press Enter.

### Verify Connection

Once connected, verify you have GPU access:
```bash
nvidia-smi
```

You should see your GPU(s) listed.

---

## Step 4: Set Up VSCode Remote SSH

VSCode Remote SSH lets you edit files on the remote server as if they were local.

### Install the Extension

1. **Open VSCode**

2. **Open Extensions** (Ctrl+Shift+X or Cmd+Shift+X)

3. **Search for** "Remote - SSH"

4. **Install** the extension by Microsoft

### Configure SSH Host

1. **Open Command Palette** (Ctrl+Shift+P or Cmd+Shift+P)

2. **Type** "Remote-SSH: Open SSH Configuration File"

3. **Select** the config file (usually `~/.ssh/config` or `C:\Users\YourName\.ssh\config`)

4. **Add your server configuration:**

   **For RunPod:**
   ```
   Host runpod-gpu
       HostName 194.68.xxx.xxx
       User root
       Port 12345
       IdentityFile ~/.ssh/id_ed25519
       StrictHostKeyChecking no
       UserKnownHostsFile /dev/null
   ```

   **For Lambda Labs:**
   ```
   Host lambda-gpu
       HostName 194.68.xxx.xxx
       User ubuntu
       Port 22
       IdentityFile ~/.ssh/id_ed25519
   ```

   **For Vast.ai:**
   ```
   Host vastai-gpu
       HostName ssh.vast.ai
       User root
       Port 12345
       IdentityFile ~/.ssh/id_ed25519
   ```

5. **Save the file**

### Connect to Remote Server

1. **Open Command Palette** (Ctrl+Shift+P)

2. **Type** "Remote-SSH: Connect to Host"

3. **Select** your configured host (e.g., `runpod-gpu`)

4. **Wait** for VSCode to set up the remote environment (first time takes a few minutes)

5. **Open a folder** on the remote server:
   - File → Open Folder
   - Navigate to `/root/nanochat` (or wherever you cloned it)

### Install Extensions on Remote

Some extensions need to be installed on the remote server:

1. Go to Extensions (Ctrl+Shift+X)
2. Look for extensions marked "Install in SSH: hostname"
3. Install Python, Pylance, and any others you need

---

## Step 5: Port Forwarding for Web UI

When running nanochat's web server, you need to forward the port to access it locally.

### Method 1: VSCode Port Forwarding (Easiest)

1. **Start the web server** on the remote:
   ```bash
   python -m scripts.chat_web
   ```

2. **VSCode automatically detects** the port and shows a notification

3. **Click "Open in Browser"** or go to the **Ports** panel (bottom of VSCode)

4. **If not automatic:**
   - Open Command Palette (Ctrl+Shift+P)
   - Type "Forward a Port"
   - Enter `8000`

5. **Open** `http://localhost:8000` in your browser

### Method 2: SSH Command Line Forwarding

**Forward port 8000:**
```bash
# Run this from your LOCAL terminal (not the remote!)
ssh -L 8000:localhost:8000 root@194.68.xxx.xxx -p 12345
```

**What `-L 8000:localhost:8000` means:**
- First `8000`: Local port (on your computer)
- `localhost`: The remote's perspective of itself
- Second `8000`: Remote port (where nanochat runs)

**Keep this terminal open** while using the web UI.

### Method 3: SSH Config with Port Forwarding

Add to your `~/.ssh/config`:
```
Host runpod-gpu
    HostName 194.68.xxx.xxx
    User root
    Port 12345
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 8000 localhost:8000
    LocalForward 6006 localhost:6006   # For TensorBoard
```

Now every time you connect, ports are automatically forwarded.

---

## Provider-Specific Instructions

### RunPod Complete Setup

```bash
# 1. Connect to your pod
ssh root@194.68.xxx.xxx -p 12345

# 2. Clone nanochat
cd ~
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# 3. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal

# 4. Set up environment
uv sync --extra gpu

# 5. Activate environment
source .venv/bin/activate

# 6. Verify GPU access
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# 7. Run the speedrun (full training)
bash runs/speedrun.sh
```

### Lambda Labs Complete Setup

```bash
# 1. Connect
ssh ubuntu@194.68.xxx.xxx

# 2. Clone nanochat
cd ~
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# 3. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 4. Set up environment
uv sync --extra gpu

# 5. Activate and verify
source .venv/bin/activate
nvidia-smi

# 6. Start training
bash runs/speedrun.sh
```

### Vast.ai Complete Setup

```bash
# 1. Connect (check your instance for exact command)
ssh -p 12345 root@ssh.vast.ai

# 2. Some Vast.ai instances need updates
apt update && apt install -y git curl

# 3. Clone nanochat
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# 4. Install uv and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync --extra gpu

# 5. Activate and run
source .venv/bin/activate
bash runs/speedrun.sh
```

---

## Troubleshooting

### "Connection refused" or "Connection timed out"

**Causes:**
- Server not running
- Wrong IP address or port
- Firewall blocking connection

**Solutions:**
1. Check if your instance is running in the provider dashboard
2. Verify IP and port are correct
3. Try restarting the instance

### "Permission denied (publickey)"

**Causes:**
- SSH key not added to provider
- Wrong key file specified

**Solutions:**
1. Verify your public key is added to the provider
2. Check `IdentityFile` path in SSH config
3. Try with explicit key:
   ```bash
   ssh -i ~/.ssh/id_ed25519 root@194.68.xxx.xxx -p 12345
   ```

### "Host key verification failed"

**Cause:** Server's identity changed (common with cloud instances)

**Solution:**
```bash
# Remove old key
ssh-keygen -R "[194.68.xxx.xxx]:12345"

# Or for default port
ssh-keygen -R "194.68.xxx.xxx"

# Then connect again
```

### VSCode "Could not establish connection"

**Solutions:**
1. Try connecting via terminal first to verify SSH works
2. Check Remote-SSH output: View → Output → Remote-SSH
3. Kill any stuck remote processes:
   ```bash
   # From terminal
   ssh your-host "pkill -f vscode-server"
   ```

### Port forwarding not working

**Solutions:**
1. Check if the port is in use locally:
   ```bash
   # On macOS/Linux
   lsof -i :8000

   # On Windows
   netstat -ano | findstr :8000
   ```

2. Try a different local port:
   ```bash
   ssh -L 8001:localhost:8000 root@194.68.xxx.xxx -p 12345
   # Then access http://localhost:8001
   ```

3. Make sure nanochat server is running on the remote

### Disconnected during long training

**Solutions:**
1. Use `screen` or `tmux` to keep processes running:
   ```bash
   # Start a screen session
   screen -S training

   # Run your command
   bash runs/speedrun.sh

   # Detach: Press Ctrl+A, then D

   # Reattach later
   screen -r training
   ```

2. Configure SSH keep-alive in `~/.ssh/config`:
   ```
   Host *
       ServerAliveInterval 60
       ServerAliveCountMax 3
   ```

---

## Tips for Efficient Remote Development

### Use `screen` or `tmux` for Long-Running Tasks

```bash
# Start a new screen session
screen -S nanochat

# Run your training
bash runs/speedrun.sh

# Detach (Ctrl+A, then D)

# List sessions
screen -ls

# Reattach
screen -r nanochat
```

### Sync Files with `rsync`

```bash
# Upload local files to remote
rsync -avz --progress ./my_data/ runpod-gpu:~/nanochat/data/

# Download results from remote
rsync -avz --progress runpod-gpu:~/nanochat/checkpoints/ ./local_checkpoints/
```

### Use VSCode Tasks

Create `.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Train Base Model",
            "type": "shell",
            "command": "source .venv/bin/activate && python -m scripts.base_train --depth=12",
            "group": "build"
        },
        {
            "label": "Start Web UI",
            "type": "shell",
            "command": "source .venv/bin/activate && python -m scripts.chat_web",
            "group": "build"
        }
    ]
}
```

### Monitor GPU Usage

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use nvitop (if installed)
pip install nvitop
nvitop
```

### Estimate Costs Before Starting

| Task | Time (8×H100) | Approx. Cost |
|------|---------------|--------------|
| Tokenizer training | 5 min | ~$2 |
| Data download | 10 min | ~$4 |
| Pretraining (d24) | 3 hours | ~$75 |
| SFT | 30 min | ~$12 |
| **Total speedrun** | **~4 hours** | **~$100** |

### Shut Down When Done!

Cloud GPUs charge by the hour. Always stop or terminate your instance when finished:

- **RunPod**: Click "Stop" or "Terminate" on your pod
- **Lambda Labs**: Click "Terminate" on your instance
- **Vast.ai**: Click "Destroy" on your instance

---

## Quick Reference

### SSH Commands Cheat Sheet

| Task | Command |
|------|---------|
| Connect | `ssh user@host -p port` |
| Connect with config | `ssh hostname` |
| Forward port | `ssh -L local:localhost:remote user@host` |
| Copy file to remote | `scp file.txt user@host:~/` |
| Copy file from remote | `scp user@host:~/file.txt ./` |
| Copy folder | `scp -r folder/ user@host:~/` |
| Remove host key | `ssh-keygen -R host` |

### VSCode Shortcuts

| Task | Shortcut |
|------|----------|
| Command Palette | Ctrl+Shift+P (Cmd+Shift+P) |
| Open Remote | Ctrl+Shift+P → "Remote-SSH: Connect" |
| Forward Port | Ctrl+Shift+P → "Forward a Port" |
| View Ports | Ctrl+Shift+P → "Ports: Focus" |
| Terminal | Ctrl+` |

---

*This guide is part of the nanochat documentation. See also: [Quick Start](QUICK_START.md), [User Guide](USER_GUIDE.md)*
