# re-use the ssh key connected to githun
mkdir -p ~/.ssh
ln -sf /data/tanyagoyal/.ssh/id_ed25519 ~/.ssh/id_ed25519
ln -sf /data/tanyagoyal/.ssh/id_ed25519.pub ~/.ssh/id_ed25519.pub
chmod 700 ~/.ssh
# avoid interactive "authenticity of host" prompt (no TTY in k8s job)
ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null

# re-use the huggingface auth
mkdir -p ~/.cache/huggingface
ln -sf /data/tanyagoyal/.huggingface/token ~/.cache/huggingface/token
hf auth whoami


git clone --branch feat/abc_methods --single-branch \
  git@github.com:tagoyal/abstention-reasoning.git
cd abstention-reasoning/

pip uninstall -y torchao


pip install -e . 
pip install -e verl/

#hf download tanyagoyal-p/abstention-reasoning-data   --repo-type dataset   --local-dir /data/tanyagoyal/artifacts
ln -sfn /data/tanyagoyal/artifacts artifacts