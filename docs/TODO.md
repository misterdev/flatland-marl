### Fixed speed 1
- [ ] evaluate 1k episodes
- [ ] train 1k episodes 

### Backlog
- [ ] Permutation of binaries in bitmap relative to the current train
- [ ] return observation from env.step
- [ ] Handle malfunctions

### TODO
- [ ] Calculate altpaths once
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