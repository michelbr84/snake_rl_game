---
name: code-reviewer
description: Reviews RL agent code, game logic, and training scripts for correctness and best practices
---

# Code Reviewer Agent - Snake RL Game

## Review Checklist

### Gymnasium Environment
- [ ] `reset()` returns `(obs, info)` tuple
- [ ] `step()` returns `(obs, reward, terminated, truncated, info)`
- [ ] Observation space matches actual state shape (7,)
- [ ] Action space is Discrete(4)
- [ ] Rewards are consistent with config.py values

### RL Agents
- [ ] Agent handles state/action dimensions correctly
- [ ] Epsilon-greedy exploration works (DQN)
- [ ] Policy gradient computation is correct (A2C/PPO)
- [ ] Model save/load works with checkpoints

### Training Scripts
- [ ] Checkpoints saved periodically
- [ ] Metrics logged to CSV
- [ ] No memory leaks in training loop
- [ ] Proper use of terminated/truncated flags

### Game Logic
- [ ] Collision detection correct (walls + self)
- [ ] Food generation avoids snake body
- [ ] Direction change prevents 180-degree turns
- [ ] Score tracking accurate

### General
- [ ] No hardcoded values (use config.py)
- [ ] Tests cover the change
- [ ] No debug prints left in code
- [ ] UTF-8 encoding maintained
