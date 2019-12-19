#!/bin/bash
clear
echo "Downloading runs/ directory from invariant01"

rsync -vaz --progress --delete --force --ignore-times --exclude '*.code-workspace' \
-e 'ssh -i ~/.ssh/FilippoVajanaOrobix_rsa' \
orobix@192.168.3.77:/home/orobix/Documents/fvajana/thesis/code/runs/. './runs'
