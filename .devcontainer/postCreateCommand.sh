# setup bash-completion & docker-completion
# cf. https://docs.docker.com/engine/cli/completion/#bash
sudo apt install bash-completion
mkdir -p ~/.local/share/bash-completion/completions
docker completion bash > ~/.local/share/bash-completion/completions/docker

# setup git-completion
curl -o ~/.git-completion.sh https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash
cat >> ~/.bashrc <<EOF

# setup git-completion
if [ -f ~/.git-completion.sh ]; then
  . ~/.git-completion.sh
fi
EOF

# Install git-secret
sudo apt-get update
sudo apt-get install -y git-secrets
git secrets --install
git secrets --register-aws
git secrets --add 'sk-ant-[a-zA-Z0-9-]+'
git secrets --add 'sk-proj-[a-zA-Z0-9-]+'