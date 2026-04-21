#!/bin/bash
set -e
/app/scripts/entrypoint.sh &
BACKEND_PID=$!
nginx -g "daemon off;" &
NGINX_PID=$!
wait $BACKEND_PID $NGINX_PID
