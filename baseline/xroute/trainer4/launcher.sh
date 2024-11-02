#!/usr/bin/env bash

# This script is used to launch the trainer4s.

if [ $# -eq 0 ]; then
  echo "Usage: $0 <number_of_trainers>"
  exit 1
fi

for ((i=0; i<$1; i++)); do
  echo "Launching t4$i"
  screen -dmS t4$i python3 main.py $((i+6667)) $((i+5556)) 1
done
