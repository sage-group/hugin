#!/bin/bash

. /home/hugin/venv/bin/activate
echo $@
exec /home/hugin/venv/bin/hugin $@