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

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

aws s3 sync s3://ar-bird-vocalization-dataset/pretrain-data/ data/