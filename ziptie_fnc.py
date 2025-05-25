# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "myrtle",
# ]
#
# [tool.uv.sources]
# myrtle = { path = "../myrtle", editable = true }
# ///
import os
import time
from sqlogging import logging
from myrtle import bench
from myrtle.config import log_directory
from myrtle.agents.fnc_ziptie_one_step import FNCZiptieOneStep
from myrtle.worlds.pendulum_discrete import PendulumDiscrete

curiosity_scale = 1.0
exploitation_factor = 2.0
feature_decay_rate = 1.0
learning_rate = 0.03
n_features = 3000
trace_decay_rate = 0.3
ziptie_threshold = 3.0

n_loop_steps = 1e6
n_episodes = 1
loops_per_second = 8
speedup = 8
verbose = True
fnc_snapshot_flag = True
fnc_snapshot_interval = 1000
ziptie_snapshot_flag = True
ziptie_snapshot_interval = 1100

db_name = f"pendulum_discrete_fnc_{int(time.time())}"


def main():
    start_time = time.time()

    bench.run(
        FNCZiptieOneStep,
        PendulumDiscrete,
        log_to_db=True,
        logging_db_name=db_name,
        world_args={
            "n_loop_steps": n_loop_steps,
            "n_episodes": n_episodes,
            "loop_steps_per_second": loops_per_second,
            "speedup": speedup,
            "verbose": verbose,
        },
        agent_args={
            "curiosity_scale": curiosity_scale,
            "exploitation_factor": exploitation_factor,
            "feature_decay_rate": feature_decay_rate,
            "n_features": n_features,
            "reward_update_rate": learning_rate,
            "trace_decay_rate": trace_decay_rate,
            "ziptie_threshold": ziptie_threshold,
            "fnc_snapshot_flag": fnc_snapshot_flag,
            "fnc_snapshot_interval": fnc_snapshot_interval,
            "ziptie_snapshot_flag": ziptie_snapshot_flag,
            "ziptie_snapshot_interval": ziptie_snapshot_interval,
        },
    )

    run_time = time.time() - start_time
    print()
    print(f"Ran in {int(run_time)} seconds")

    logger = logging.open_logger(
        name=db_name,
        dir_name=log_directory,
        level="info",
    )
    result = logger.query(
        f"""
        SELECT AVG(reward)
        FROM {db_name}
        GROUP BY episode
        ORDER BY episode DESC
    """
    )
    for i_episode in range(len(result)):
        print(f"Episode {i_episode} average reward: {result[i_episode][0]}")
    print()

    db_filename = f"{db_name}.db"
    db_path = os.path.join(log_directory, db_filename)
    os.remove(db_path)


if __name__ == "__main__":
    main()
