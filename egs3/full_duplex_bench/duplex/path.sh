#!/bin/bash

export PYTHONPATH=../../../:../../TEMPLATE/duplex:$(pwd):${PYTHONPATH:-}

source ../../../tools/activate_python.sh
source ../../../tools/extra_path.sh
