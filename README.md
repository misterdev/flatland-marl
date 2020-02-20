# Solution the Flatland Challenge

# Usage
## Rendering
```bash
python src/main.py --render
```

## Training
```bash
python src/main.py --train --num-episodes=10000 --prediction-depth=150 --eps=0.9998 --checkpoint-interval=100 --buffer-size=10000
```

## Visualize
```
tensorboard --logdir=runs
```