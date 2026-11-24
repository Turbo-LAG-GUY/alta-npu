Update from Jordan's desktop session: I stopped your alta-llm-prefill2 run at 17:24 because the board was thrashing swap. Please record this in the project docs.

What happened
- 15:32 alta-llm-prefill2 (llm/l3/prefill2-ladder.sh) started. Q1, Q2 and Q3 (135m-last) all passed, rc=0, kernel_warnings=0, about 1.5 min each.
- 15:37 to 17:24: no output for 1h47m. The board was stuck, not computing.
- At 17:20 the board had load around 33, 193 MB RAM available out of 3.7 GB, 4 GB swap in use, and 2.2 GB "shared" memory. /dev/shm and the other tmpfs mounts were empty, so the shared memory was most likely the job's NPU/DRM buffers.
- 25 tasks were in D state waiting on block I/O: kswapd0, Xorg, both gnome-shells, every claude process, and the jb_prefill worker. That is swap thrashing, not an NPU hang.
- systemd recorded the run as: 8m53s CPU, 1.5 GB memory peak, 407.8 MB swap peak.

What I did (Jordan approved it)
1. 17:24:21 `systemctl --user stop alta-llm-prefill2`. It stopped cleanly at 17:24:52. You didn't cause the failure: the run was interrupted from outside because it had stalled.
2. Sent SIGTERM to three duplicate `claude -r` processes (186020, 186099, 186166), which had been idle for about 17h. All three exited. I left your session, its daemon and the `claude -r` that spawned it (185911) running, plus `claude` 1057538.
3. Afterwards: 1.2 GB RAM available, shared memory down to 708 MB, swap down by about 1 GB. Load is still high while pages swap back in.

Since then
- You restarted the job at 17:26:38. Q1 passed again at 17:28:55 (compile 120 steps in 6989 ms, 787.3 ms per chunk), so it's progressing with the memory freed.

Why this belongs in the docs
- This ladder needs about 1.5 GB peak plus NPU buffers. On a 3.7 GB A311D board that also runs two GNOME sessions and Xorg, it gets pushed into swap after Q3 and stalls without erroring. A stall shows up as silence, not a failure.
- Suggested notes: (a) memory requirement and peak per step; (b) run without a desktop session logged in, or close one of the two GNOME sessions first; (c) have the wait loop treat "no new output for N minutes" as a failure, not keep waiting; (d) don't keep extra `claude -r` resumes open on the board.
- If Q4 and later steps stall again at the same point, check `free -m` and the D-state count before blaming the NPU driver.
