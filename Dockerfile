FROM nvidia/cuda:11.6.2-base-ubuntu20.04

RUN rm -rf /var/lib/apt/lists/* && \
    apt-get -y -q update && \
    apt-get -y -q upgrade && \
    apt-get -y -q install python3-pip g++
WORKDIR /workspace
COPY . .
RUN pip3 install torch gym pandas wandb && \
    g++ -shared ./environment/c/sudoku_score.cpp -o ./environment/c/sudoku.so

CMD ["python3", "train.py"]