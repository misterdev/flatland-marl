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
        - [x] same as above, in general, if an agent moves to the next cell occupied by another agent with id > it doesn't move (this should be already handled by delay)
        - [x] train3 stops in front of train0. Train0 moves first, it makes a choice, it decides to move_forward. Then train3 moves, it decides to stop, it delays Train0, but train0 has already made a choice. That choice will remain valid until train3 moves away, at the next step train0 will move_f and action_required will be True. This consumes 1 bit in train0 bitmap.
    - [x] check what delay does to prediction
    - [x] switching to action_required strategy makes bitmap lose sync, assert path.len * speed == bitmap.len


- [ ] Implement strategy to apply the action before entering last cell before a switch
- [ ] Re-enable different train speeds
- [ ] Handle malfunctions


# REASONING
- [ ] delay should stop agent
- [x] agent can be started only if its initial position is empty