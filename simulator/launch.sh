#!/bin/bash

set -euo pipefail

docker run -it --rm -v $PWD:/app -w /app -p 8080:8080 xplanlab/xroute-env
