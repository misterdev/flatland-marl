# Solution the Flatland Challenge

# Usage
### Train
```bash
python src/main.py --train --num-episodes=10000 --prediction-depth=150 --eps=0.9998 --checkpoint-interval=100 --buffer-size=10000
```

### Using tensorboard
```bash
tensorboard --logdir=runs
```

### Parameters
#### Rendering
```bash
python src/main.py --render
```
### Plotting
```bash
python src/main.py --plot
```