### Fixed speed 1
- [ ] evaluate 1k episodes
- [ ] train 1k episodes 

### Backlog
- [ ] Permutation of binaries in bitmap relative to the current train
- [ ] return observation from env.step
- [ ] Handle malfunctions

### TODO
- [ ] Calculate altpaths once
- [ ] giulia's error. NOTE: it has to do with prediction depth
- [ ] FIX
    ```
        Traceback (most recent call last):
    File "src/main.py", line 370, in <module>
        main(args)
    File "src/main.py", line 197, in main
        best_i = np.random.choice(np.arange(len(altmaps)))
    File "mtrand.pyx", line 773, in numpy.random.mtrand.RandomState.choice
    ValueError: 'a' cannot be empty unless no samples are taken
    ```
- [ ] FIX
    ```
    4 Agents on (20,20). Episode: 949        Mean done agents: 0.23  Mean reward: -372.12    Mean normalized reward: -0.77   Epsilon: 0.15
    [WARN] agent's 0 path run out
    [WARN] agent's 0 path run out
    [WARN] agent's 0 path run out
    Traceback (most recent call last):
    File "src/main.py", line 379, in <module>
    File "src/main.py", line 194, in main
        best_i = argmax // 2
    File "<__array_function__ internals>", line 6, in argmax
    File "/home/devid/anaconda3/envs/flatcla/lib/python3.6/site-packages/numpy/core/fromnumeric.py", line 1153, in argmax
        return _wrapfunc(a, 'argmax', axis=axis, out=out)
    File "/home/devid/anaconda3/envs/flatcla/lib/python3.6/site-packages/numpy/core/fromnumeric.py", line 61, in _wrapfunc
        return bound(*args, **kwds)
    ValueError: attempt to get argmax of an empty sequence
    ```


# Results
- No train
```
4 Agents on (20,20). Episode: 937        Mean done agents: 0.23  Mean reward: -365.44    Mean normalized reward: -0.76   Done agents in last episode: 0.0
4 Agents on (20,20). Episode: 938        Mean done agents: 0.23  Mean reward: -365.89    Mean normalized reward: -0.76   Done agents in last episode: 0.0
4 Agents on (20,20). Episode: 939        Mean done agents: 0.23  Mean reward: -368.21    Mean normalized reward: -0.76   Done agents in last episode: 0.0
4 Agents on (20,20). Episode: 940        Mean done agents: 0.23  Mean reward: -369.19    Mean normalized reward: -0.76   Done agents in last episode: 0.2
4 Agents on (20,20). Episode: 941        Mean done agents: 0.23  Mean reward: -371.35    Mean normalized reward: -0.77   Done agents in last episode: 0.0
4 Agents on (20,20). Episode: 942        Mean done agents: 0.23  Mean reward: -370.29    Mean normalized reward: -0.77   Done agents in last episode: 0.2
4 Agents on (20,20). Episode: 943        Mean done agents: 0.23  Mean reward: -369.51    Mean normalized reward: -0.76   Done agents in last episode: 0.0
4 Agents on (20,20). Episode: 944        Mean done agents: 0.22  Mean reward: -367.83    Mean normalized reward: -0.76   Done agents in last episode: 0.0
4 Agents on (20,20). Episode: 945        Mean done agents: 0.22  Mean reward: -368.65    Mean normalized reward: -0.76   Done agents in last episode: 0.0
4 Agents on (20,20). Episode: 946        Mean done agents: 0.23  Mean reward: -367.46    Mean normalized reward: -0.76   Done agents in last episode: 0.5
4 Agents on (20,20). Episode: 947        Mean done agents: 0.23  Mean reward: -368.27    Mean normalized reward: -0.76   Done agents in last episode: 0.2
4 Agents on (20,20). Episode: 948        Mean done agents: 0.23  Mean reward: -371.35    Mean normalized reward: -0.77   Done agents in last episode: 0.5
4 Agents on (20,20). Episode: 949        Mean done agents: 0.23  Mean reward: -372.12    Mean normalized reward: -0.77   Done agents in last episode: 0.5
4 Agents on (20,20). Episode: 949        Mean done agents: 0.23  Mean reward: -372.12    Mean normalized reward: -0.77   Epsilon: 0.15
```

- Train 1k
```
4 Agents on (20,20). Episode: 997        Mean done agents: 0.46  Mean reward: -290.34    Mean normalized reward: -0.60   Done agents in last episode: 0.50%      Epsilon: 0.8
4 Agents on (20,20). Episode: 998        Mean done agents: 0.46  Mean reward: -288.93    Mean normalized reward: -0.60   Done agents in last episode: 0.75%      Epsilon: 0.8
4 Agents on (20,20). Episode: 999        Mean done agents: 0.47  Mean reward: -286.31    Mean normalized reward: -0.59   Done agents in last episode: 0.75%      Epsilon: 0.8
4 Agents on (20,20). Episode: 999        Mean done agents: 0.47  Mean reward: -286.31    Mean normalized reward: -0.59   Epsilon: 0.82
```

[WARN] agent's 0 path run out
[WARN] agent's 0 path run out
Traceback (most recent call last):
  File "src/main.py", line 371, in <module>
    main(args)
  File "src/main.py", line 194, in main
    best_i = np.random.choice(np.arange(len(altmaps)))
  File "mtrand.pyx", line 773, in numpy.random.mtrand.RandomState.choice
ValueError: 'a' cannot be empty unless no samples are taken