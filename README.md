# ERASMUS GPT

Here we find the adjusted version of [lit-gpt](https://github.com/Lightning-AI/lit-gpt) to be used on our cluster


## Prepare

``` shell
wget https://nvidia.github.io/nvidia-docker/gpgkey --no-check-certificate
sudo apt-key add gpgkey
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

To provide access I also need:

``` shell
sudo gpasswd -a ffok docker
sudo chown ffok /var/run/docker.sock
```

If you are going to use Huggingface Datasets:

``` shell
apt-get update && apt-get install -y sudo curl git && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && sudo apt-get install git-lfs
```

## Build

This will take a while!!!

``` shell
git clone https://github.com/FlorisFok/lit-gpt-erasmus.git
docker build -t erasmus-gpt:0.2.2 --network host lit-gpt-erasmus/
```


## Download assets

Get inside! From here on, we provide command execute from within the docker (unless state elsewise)

``` shell
docker run -v /home/ffok/data:/app/data -v /home/ffok/out:/app/out -v /home/ffok/checkpoints:/app/checkpoints --network host --gpus all -it erasmus-gpt:0.2.2 bash
```

If you are planning to download huggingface datasets, please also install:

``` shell
apt-get update && apt-get install -y sudo curl git && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && sudo apt-get install git-lfs
```

To download the desired model, run the following:

``` shell
export MY_MODEL=EleutherAI/pythia-1b
python3 scripts/download.py --repo_id $MY_MODEL
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MY_MODEL
```

## Prepare data

Data needs to be placed in a folder, the folder contains one or more `.jsonl` with format:
`{"text": <text>}\n` for each row.  
  
Now lets Tokenize! (9min / gb approx.)

``` shell
python3 scripts/prepare_any.py --checkpoint_dir data/$MY_MODEL --destination_path data/$OUT_FILE --data_dir data/$IN_FILE
```


## Run MultiNode

Here we are on the machine, not in the docker, and we are logged into two machines. For node1 one we have:

``` shell
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ffok/data:/app/data -v /home/ffok/out:/app/out -v /home/ffok/lit-gpt-erasmus/pretrain:/app/pretrain -v /home/ffok/checkpoints:/app/checkpoints -e NCCL_IB_DISABLE=1 --network host -v /home/ffok/lit-gpt-erasmus/lit_gpt:/app/lit_gpt --gpus all -d erasmus-gpt:0.2.2 lightning run model --node-rank=0 --main-address=10.3.123.10 --accelerator=cuda --devices=4 --num-nodes=2 pretrain/trainer2.py --train_data_dir data/$OUT_FILE --pretrain checkpoints/$MY_MODEL
```

And node 2:

``` shell
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ffok/data:/app/data -v /home/ffok/out:/app/out -v /home/ffok/lit-gpt-erasmus/pretrain:/app/pretrain -v /home/ffok/checkpoints:/app/checkpoints -e NCCL_IB_DISABLE=1 --network host -v /home/ffok/lit-gpt-erasmus/lit_gpt:/app/lit_gpt --gpus all -d erasmus-gpt:0.2.2 lightning run model --node-rank=1 --main-address=10.3.123.10 --accelerator=cuda --devices=4 --num-nodes=2 pretrain/trainer2.py --train_data_dir data/$OUT_FILE --pretrain checkpoints/$MY_MODEL
```

!! For this to work we need to have both machines connected by SSH. Meaning they can ssh between each other without passwords !!