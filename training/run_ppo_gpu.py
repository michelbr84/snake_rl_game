"""Script para treinar PPO com GPU e sem renderização."""
import sys
import os

# Garantir que roda do diretório raiz do projeto
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# Configurar SDL antes de importar pygame
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Forçar flush imediato
sys.stdout.reconfigure(line_buffering=True)

# Suprimir prints verbosos do jogo
_original_print = print
_suppress = ['Comida gerada', 'Jogo resetado', 'Jogo iniciado', 'Nova itera',
             'Dire', 'Nova posi', 'Colis', 'Comida consumida']

def quiet_print(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    if any(p in msg for p in _suppress):
        return
    _original_print(*args, **kwargs, flush=True)

import builtins
builtins.print = quiet_print

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
_original_print(f"Training PPO on: {gpu_name} ({device})", flush=True)

sys.path.insert(0, os.getcwd())
from training.train_ppo import train_ppo
train_ppo()
