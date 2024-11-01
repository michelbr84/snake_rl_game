from setuptools import setup, find_packages

setup(
    name='snake_rl_game',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gym',
        'torch',
        'numpy',
        'opencv-python',
        'pygame',
        'tensorboard',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'run_game=run_game:main',
        ],
    },
    author='Michel Duek',
    description='Projeto de Jogo Snake com IA de Aprendizado por Reforço',
)