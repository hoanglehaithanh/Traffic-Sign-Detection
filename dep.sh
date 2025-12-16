#!/bin/bash

# Update package database
echo "Updating package database..."
sudo pacman -Syu

# Remove conflicting packages
echo "Removing conflicting packages..."
sudo pacman -R --noconfirm gst-plugin-gtk gst-plugins-bad-libs

# Install the target versions
echo "Installing GStreamer and related packages..."
sudo pacman -S --noconfirm gstreamer gst-plugins-base-libs gst-plugin-gtk gst-plugins-bad-libs

echo "completed."
