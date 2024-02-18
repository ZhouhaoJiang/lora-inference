#!/bin/bash

# Step 1: Clone the repository
git clone https://github.com/ZhouhaoJiang/lora-inference.git

# Change directory to the cloned repository
cd lora-inference

# Step 2: Checkout to the specific branch
git checkout with-lora-and-mask

# Step 3: Install Cog
curl https://replicate.github.io/codespaces/scripts/install-cog.sh | bash

# Step 4: Install Docker
# Note: The following commands are for Debian-based systems like Ubuntu.
# Update your package lists
sudo apt-get update

# Install prerequisite packages which let apt use packages over HTTPS
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# Add the GPG key for the official Docker repository to your system
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add the Docker repository to APT sources
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Update your package lists with the new repository
sudo apt-get update

# Install Docker
sudo apt-get install -y docker-ce

# Add your user to the docker group (replace $USER with your username if necessary)
sudo usermod -aG docker $USER

# Inform the user that a reboot or re-login is needed
echo "Docker installation complete. You may need to reboot or log out & log back in for the 'docker' command to work."

# End of script
