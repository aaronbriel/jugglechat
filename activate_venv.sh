#!/bin/bash

VIRTUAL_ENV=$1
if [ -z $VIRTUAL_ENV ]; then
  echo "usage: $0 </home/aaron/jugglechat/.venv/> <cmd>"
  exit 1
fi

. $VIRTUAL_ENV/bin/activate
shift 1
exec "$@"
deactivate