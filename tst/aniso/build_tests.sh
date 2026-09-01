#!/usr/bin/env bash
# Spanning compile matrix for the aniso pipeline.
# NEEDS RUNNING IN TOP DIRECTORY
set -u

PROB=aniso-tests
FLUX=hlle_mod
LOG=$(pwd)/build_matrix.log
: > "$LOG"

# Each entry is a full configure argument string.
CONFIGS=(
  "-b --prob=$PROB --flux=$FLUX"
  "-b --prob=$PROB --flux=$FLUX -sts"
  "-b --prob=$PROB --flux=$FLUX -mpi"
  "-b --prob=$PROB --flux=$FLUX -mpi -sts"
  "-b --prob=$PROB --flux=$FLUX -omp"
  "-b --prob=$PROB --flux=$FLUX -mpi -omp -sts"
  "-b --prob=$PROB --flux=$FLUX -float"
  "-b --prob=$PROB --flux=$FLUX -float -sts"
  "-b --prob=$PROB --flux=$FLUX -debug"
  "-b --prob=$PROB --flux=$FLUX -debug -sts"
  "-b --prob=$PROB --flux=$FLUX -hdf5"
  "-b --prob=$PROB --flux=$FLUX -hdf5 -mpi -sts"
  "-b --prob=$PROB --flux=$FLUX --nghost=4"
  "-b --prob=$PROB --flux=$FLUX --nghost=4 -sts"
  "--prob=$PROB --flux=hlle"
  "--prob=$PROB --flux=hlle -sts"
)

PASS=0; FAIL=0
for cfg in "${CONFIGS[@]}"; do
  echo "=================================================================" | tee -a "$LOG"
  echo "CONFIG: $cfg" | tee -a "$LOG"
  make clean >/dev/null 2>&1
  if ! python configure.py $cfg >> "$LOG" 2>&1; then
    echo "  CONFIGURE FAILED" | tee -a "$LOG"; FAIL=$((FAIL+1)); continue
  fi
  if make -j"$(nproc)" >> "$LOG" 2>&1; then
    W=$(grep -c "warning:" "$LOG")
    echo "  BUILD OK  (cumulative warnings: $W)" | tee -a "$LOG"
    PASS=$((PASS+1))
  else
    echo "  BUILD FAILED" | tee -a "$LOG"; FAIL=$((FAIL+1))
  fi
done

echo "=================================================================" | tee -a "$LOG"
echo "passed: $PASS   failed: $FAIL" | tee -a "$LOG"
NWARN=$(grep -c "warning:" "$LOG")
echo "warnings: $NWARN" | tee -a "$LOG"
grep -n "warning:" "$LOG" | head -50
