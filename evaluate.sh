#!/bin/bash

perf stat ./mc_growth.exe 1>$1 2>&1
perf record ./mc_growth.exe
mv perf.data ${1}_perf.data
