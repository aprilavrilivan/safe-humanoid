# Within-task PPO evaluation figures

Four task dashboards compare Clean, Time-domain, and Frequency-domain PPO using
the same seven metrics. Every bar is the mean across 3 training seeds, error bars
show ±1 seed SD, and dots show the individual seeds. Each seed was evaluated for
20 episodes in the same task used for training.

- `nominal_metrics.png`
- `aggressive_metrics.png`
- `abrupt_metrics.png`
- `push_recovery_metrics.png`

Metrics: termination rate, XY and yaw tracking error, policy-rate torque RMS,
200 Hz mean estimated mechanical power, and 200 Hz torque/joint-speed high-band
(25–100 Hz) power.

Important interpretation limits:

- Safety quantities are simulated actuator-loading proxies. Hardware torque and
  speed limits were not calibrated.
- There are 3 independent training seeds, but only one evaluation seed. The 20
  episodes per evaluation are not independent algorithm-level samples.
- Failed episodes are shorter and contribute fewer samples to mean torque and
  power metrics. Inspect termination and duration alongside those metrics.
- Raw 200 Hz traces were not retained locally, so event-aligned time histories,
  full PSD curves, recovery time, and overshoot cannot be reconstructed.

Regenerate from the repository root with:

```bash
python scripts/plot_reward_ppo_results.py   --archive outputs/archives/reward-ppo-v1-20260922/data   --output figs
```
