#!/usr/bin/env bash

# GW190701_203306_data0_1246048404-56_generation
# PARENTS 
# CHILDREN GW190701_203306_data0_1246048404-56_sampling
if [[ "GW190701_203306_data0_1246048404-56_generation" == *"$1"* ]]; then
    echo "Running: /home/deathstar/miniconda3/envs/dingorep/bin/dingo_pipe_generation events/GW190701_203306/H1/GW190701_203306_config_complete.ini --reuse-event-data true --label GW190701_203306_data0_1246048404-56_generation --idx 0 --trigger-time 1246048404.56 --outdir events/GW190701_203306/H1"
    /home/deathstar/miniconda3/envs/dingorep/bin/dingo_pipe_generation events/GW190701_203306/H1/GW190701_203306_config_complete.ini --reuse-event-data true --label GW190701_203306_data0_1246048404-56_generation --idx 0 --trigger-time 1246048404.56 --outdir events/GW190701_203306/H1
fi

# GW190701_203306_data0_1246048404-56_sampling
# PARENTS GW190701_203306_data0_1246048404-56_generation
# CHILDREN GW190701_203306_data0_1246048404-56_sampling_plot
if [[ "GW190701_203306_data0_1246048404-56_sampling" == *"$1"* ]]; then
    echo "Running: /home/deathstar/miniconda3/envs/dingorep/bin/dingo_pipe_sampling events/GW190701_203306/H1/GW190701_203306_config_complete.ini --reuse-event-data true --label GW190701_203306_data0_1246048404-56_sampling --event-data-file events/GW190701_203306/H1/data/GW190701_203306_data0_1246048404-56_generation_event_data.hdf5 --outdir events/GW190701_203306/H1"
    /home/deathstar/miniconda3/envs/dingorep/bin/dingo_pipe_sampling events/GW190701_203306/H1/GW190701_203306_config_complete.ini --reuse-event-data true --label GW190701_203306_data0_1246048404-56_sampling --event-data-file events/GW190701_203306/H1/data/GW190701_203306_data0_1246048404-56_generation_event_data.hdf5 --outdir events/GW190701_203306/H1
fi

# GW190701_203306_data0_1246048404-56_sampling_plot
# PARENTS GW190701_203306_data0_1246048404-56_sampling
# CHILDREN 
if [[ "GW190701_203306_data0_1246048404-56_sampling_plot" == *"$1"* ]]; then
    echo "Running: /home/deathstar/miniconda3/envs/dingorep/bin/dingo_pipe_plot --label GW190701_203306_data0_1246048404-56_sampling_plot --result events/GW190701_203306/H1/result/GW190701_203306_data0_1246048404-56_sampling.hdf5 --outdir events/GW190701_203306/H1/result --corner --weights --log_probs"
    /home/deathstar/miniconda3/envs/dingorep/bin/dingo_pipe_plot --label GW190701_203306_data0_1246048404-56_sampling_plot --result events/GW190701_203306/H1/result/GW190701_203306_data0_1246048404-56_sampling.hdf5 --outdir events/GW190701_203306/H1/result --corner --weights --log_probs
fi

