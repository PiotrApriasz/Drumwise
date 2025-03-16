#!/bin/bash

# Script to build and run the Python Docker environment

# Build the docker container
echo "Building the Docker container..."
docker-compose build

# Run the container in detached mode
echo "Starting the container..."
docker-compose up -d

# Enter the container
echo "Entering the Python environment..."
echo "Type 'exit' to leave the container when done."
docker-compose exec midiconverter bash

# Ask if user wants to stop the container when they exit
read -p "Would you like to stop the container now? (y/n): " answer
if [[ $answer == "y" || $answer == "Y" ]]; then
  echo "Stopping the container..."
  docker-compose down
  echo "Container stopped."
else
  echo "Container is still running in the background."
  echo "To stop it later, run: docker-compose down"
fi 