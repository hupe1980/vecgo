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

# Install cross-compilation toolchain for x86_64 (AMD64)
echo "🔀 Installing x86_64 cross-compilation toolchain..."
sudo apt-get install -y \
    gcc-x86-64-linux-gnu \
    libc6-dev-amd64-cross


