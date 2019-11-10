FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

COPY ./docker/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . ./thesis-code