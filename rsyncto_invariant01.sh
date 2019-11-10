#!/bin/bash
clear
echo "Synchronizing with invariant01"

rsync -rh --progress --del --force --exclude 'data/' \
-e 'ssh -i ~/.ssh/orobix/FilippoVajanaOrobix_rsa' \
'./' orobix@192.168.3.77:/home/orobix/Projects/fvajana/master-degree
