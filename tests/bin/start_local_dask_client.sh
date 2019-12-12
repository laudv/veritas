#!/bin/bash

dask-scheduler --host localhost --port 30333 &
dask-worker tcp://localhost:30333 &
dask-worker tcp://localhost:30333 &
dask-worker tcp://localhost:30333 &
dask-worker tcp://localhost:30333
