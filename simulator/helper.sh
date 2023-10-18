#!/bin/bash

set -eo pipefail

_help() {
    cat <<EOF

USAGE: $0
    -s, --start      Start the simulator.
    -e, --enter      Enter the simulator container with bash.
    -k, --kill       Kill the simulator container.
    -l, --log        Show the simulator logs.
    --build          Build the simulator image.
    --compile=PATH   Compile the OpenROAD executable.
    --bash           Launch the simulator with bash.
    --dev            Launch the simulator with dev mode.
EOF
    exit "${1:-1}"
}

case "${1}" in
    -s|--start)
        docker run -d --rm -v $PWD:/app -w /app -p 8080:8080 xplanlab/xroute-env /app/start_container
        ;;

    -e|--enter)
        docker exec -it $(docker ps | grep xroute-env | awk '{print  $1}') /bin/bash
        ;;

    -k|--kill)
        docker kill $(docker ps | grep xroute-env | awk '{print  $1}')
        ;;

    -l|--log)
        docker logs -f $(docker ps | grep xroute-env | awk '{print  $1}')
        ;;

    --build)
        if [[ -z "$TIME_ZONE" ]]; then
            TIME_ZONE="Etc/UTC"
        fi
        docker build --build-arg TZ="$TIME_ZONE" -t xplanlab/xroute-env . && docker image prune -f
        ;;

    --compile=*)
        OPENROAD_PATH="$(echo $1 | sed -e 's/^[^=]*=//g')"
        if [[ -z "$OPENROAD_PATH" ]]; then
            echo "Please specify the path to OpenROAD."
            exit 1
        fi
        docker run -it --rm -v $OPENROAD_PATH:/app -w /app xplanlab/xroute-env /bin/bash -c "\
          cmake -B cmake-build-release-docker . && cmake --build cmake-build-release-docker -j \$(nproc)"
        ;;

    --bash)
        docker run -it --rm -v $PWD:/app -w /app -p 8080:8080 xplanlab/xroute-env /app/start_container --bash
        ;;

    --dev)
        docker run -it --rm -v $PWD:/app -w /app -p 8080:8080 xplanlab/xroute-env /app/start_container --dev
        ;;

    *)
        echo "unknown option: ${1}" >&2
        _help
        ;;
esac
