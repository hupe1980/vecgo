#!/bin/bash
set -e

echo "🔧 Setting up Vecgo development environment..."

# Update package lists
sudo apt-get update

# Install build dependencies for C to assembly compilation
echo "📦 Installing clang, LLVM, and binutils..."
sudo apt-get install -y \
    clang \
    llvm \
    binutils \
    build-essential

# Install just (command runner for Justfile targets)
echo "📦 Installing just..."
sudo apt-get install -y just

# Install cross-compilation toolchain for x86_64 (AMD64)
echo "🔀 Installing x86_64 cross-compilation toolchain..."
sudo apt-get install -y \
    gcc-x86-64-linux-gnu \
    libc6-dev-amd64-cross

# Install Ruby and Jekyll for documentation (GitHub Pages compatible)
echo "💎 Installing Ruby and Jekyll (GitHub Pages version)..."
sudo apt-get install -y ruby-full build-essential zlib1g-dev

# Setup Ruby gems directory (for both bash and zsh)
mkdir -p "$HOME/gems"
export GEM_HOME="$HOME/gems"
export PATH="$HOME/gems/bin:$PATH"

# Add to shell configs (zsh is default in devcontainer)
grep -q 'GEM_HOME' ~/.zshrc 2>/dev/null || {
    echo 'export GEM_HOME="$HOME/gems"' >> ~/.zshrc
    echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.zshrc
}
grep -q 'GEM_HOME' ~/.bashrc 2>/dev/null || {
    echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
    echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
}

gem install bundler
gem install github-pages


