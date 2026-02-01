# nanochat Documentation

Welcome to the nanochat documentation! This folder contains comprehensive guides for users and developers at all skill levels.

---

## Documentation Overview

| Guide | Audience | Description |
|-------|----------|-------------|
| [Quick Start](QUICK_START.md) | Everyone | 12 hands-on use cases to get you started quickly |
| [User Guide](USER_GUIDE.md) | Users | Complete guide to using nanochat |
| [Developer Guide](DEVELOPER_GUIDE.md) | Developers | How to understand and modify the codebase |
| [Architecture](ARCHITECTURE.md) | Advanced | Deep technical details of the system |
| [Remote GPU Setup](REMOTE_GPU_SETUP.md) | Everyone | SSH tunneling for cloud GPU development |

---

## Quick Links

### For New Users
Start here if you've never used nanochat before:
1. [Installation](USER_GUIDE.md#installation-step-by-step)
2. [Quick Start Use Cases](QUICK_START.md#use-case-1-chat-with-a-pre-trained-model)

### For Training Your Own Model
Learn how to train from scratch:
1. [Quick Training (CPU)](QUICK_START.md#use-case-5-train-a-tiny-model-on-your-computer)
2. [Full GPT-2 Training](QUICK_START.md#use-case-10-run-the-full-gpt-2-training-pipeline)
3. [Training Parameters](USER_GUIDE.md#training-your-own-model)

### For Developers
Understand and extend the codebase:
1. [Development Setup](DEVELOPER_GUIDE.md#development-environment-setup)
2. [Codebase Overview](DEVELOPER_GUIDE.md#codebase-overview)
3. [Adding New Features](DEVELOPER_GUIDE.md#adding-new-features)

### For Researchers
Deep dive into the architecture:
1. [Model Architecture](ARCHITECTURE.md#model-architecture)
2. [Optimization](ARCHITECTURE.md#optimization)
3. [Design Decisions](ARCHITECTURE.md#design-decisions)

### For Cloud GPU Users
Set up remote development with VSCode:
1. [SSH Key Setup](REMOTE_GPU_SETUP.md#step-1-generate-ssh-keys)
2. [Provider Setup](REMOTE_GPU_SETUP.md#step-2-set-up-your-gpu-provider) (RunPod, Lambda, Vast.ai)
3. [VSCode Remote SSH](REMOTE_GPU_SETUP.md#step-4-set-up-vscode-remote-ssh)
4. [Port Forwarding](REMOTE_GPU_SETUP.md#step-5-port-forwarding-for-web-ui)

---

## Getting Help

- **GitHub Discussions**: https://github.com/karpathy/nanochat/discussions
- **Discord #nanochat**: Real-time community support
- **DeepWiki**: https://deepwiki.com/karpathy/nanochat

---

## Contributing to Documentation

Found an error or want to improve the docs? Contributions are welcome!

1. Fork the repository
2. Make your changes in the `docs/` folder
3. Submit a pull request

Please follow the existing style and ensure examples are tested.
