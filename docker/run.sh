docker run -it --name test_test \
--mount type=bind,source="$(pwd)"/data,target=/workspace/thesis-code/data \
--mount type=bind,source="$(pwd)/runs",target=/workspace/thesis-code/runs \
--gpus all fvajana/tesi