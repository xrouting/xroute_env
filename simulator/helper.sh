#!/bin/bash

set -eo pipefail

_help() {
    cat <<EOF

USAGE: $0
    -b, --build      Builds the simulator image.
    -l, --launch     Launches the simulator container.
    -i, --inspect    Inspects the simulator container.
    --bash           Launches the simulator container with bash.
EOF
    exit "${1:-1}"
}

case "${1}" in
    -b|--build)
        if [[ -z "$TIME_ZONE" ]]; then
            TIME_ZONE="Etc/UTC"
        fi
        docker build --build-arg TZ="$TIME_ZONE" -t xplanlab/xroute-env . && docker image prune -f
        ;;

    -l|--launch)
        docker run -it --rm -v $PWD:/app -w /app -p 8080:8080 xplanlab/xroute-env
        ;;

    -i|--inspect)
        docker exec -it $(docker ps | grep xroute-env | awk '{print  $1}') /bin/bash
        ;;

    --bash)
        docker run -it --rm -v $PWD:/app -w /app -p 8080:8080 xplanlab/xroute-env /app/start_container --bash
        ;;

    *)
        echo "unknown option: ${1}" >&2
        _help
        ;;
esac
