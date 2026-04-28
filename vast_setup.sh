ssh-keygen -t rsa -b 4096 -C "hamula.dm@gmail.com"  
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

echo "================================================"
echo "Public key:"
cat ~/.ssh/id_rsa.pub
echo "================================================"

git clone git@github.com:hamuladm/ar-bird-vocalization.git
cd ar-bird-vocalization
uv sync
uv run python -c 'from datasets import load_dataset;dataset = load_dataset("DBD-research-group/BirdSet", "XCM", split="train", trust_remote_code=True)'

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
unzip /tmp/awscliv2.zip
sudo ./aws/install

aws s3 sync s3://ar-bird-vocalization-dataset/pretrain-data/ data/