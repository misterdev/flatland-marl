### Fixed speed 1
- [ ] evaluate 1k episodes
- [ ] train 1k episodes 
Check 

### Backlog
- [ ] Permutation of binaries in bitmap relative to the current train
- [ ] Replace numbers with RailenvActions
- [ ] ? return observation from env.step


### TODO
- [ ] Fix any error with trains
    - [x] add 0 at the beginning of bitmap if agent not departed
    - [x] if train 0 READY_TO_DEPART selects MOVE_FORWARD, but it's initial cell is occupied by a train (with id >) it remain in READY_TO_DEPART, but the bitmap rolls
    - [ ] same as above, in general, if an agent moves to the next cell occupied by another agent with id > it doesn't move (this should be already handled by delay)
    - [x] check what delay does to prediction
- [ ] Implement strategy to apply the action before entering last cell before a switch
- [ ] Re-enable different train speeds
- [ ] Handle malfunctions


# REASONING
- [ ] delay should stop agent
- [x] agent can be started only if its initial position is empty