# Solution to the Flatland Challenge

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

### Docs

* Devid Farinelli, [_Apprendimento con rinforzo applicato allo scheduling dei treni per la Flatland challenge_](https://amslaurea.unibo.it/20487/1/farinelli_devid_tesi.pdf).
* Giulia Cantini, [_FLATLAND: A study of Deep Reinforcement Learning methods applied to the vehicle rescheduling problem in a railway environment_](https://amslaurea.unibo.it/20412/1/thesis_giulia_cantini.pdf)