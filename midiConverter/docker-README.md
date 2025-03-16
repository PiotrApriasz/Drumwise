# Python Docker Environment for midiConverter

This document describes how to use the Docker environment for the midiConverter Python project.

## Setup

1. Make sure you have Docker and Docker Compose installed on your system.

2. Build and start the Docker container:
   ```bash
   docker-compose up -d
   ```

3. Access the Python environment:
   ```bash
   docker-compose exec midiConverter bash
   ```

## Working with the Environment

### Running Python Scripts

Once inside the container, you can run your Python scripts as usual:

```bash
python src/converters/dl_audio_to_midi_converter.py
```

### Working with Files

The Docker environment is set up with the following volume mounts:

- `./midiConverter:/app` - Your entire midiConverter directory is mounted inside the container at `/app`
- `./data:/app/data` - The data directory is shared between your host and the container

This means:
- Any changes you make to files in midiConverter on your host machine will be immediately available in the container
- Files saved to the data directory will be accessible from both your host machine and the container

### Example Workflow

1. Start the container:
   ```bash
   docker-compose up -d
   ```

2. Enter the container:
   ```bash
   docker-compose exec midiConverter bash
   ```

3. Run your Python code:
   ```bash
   # Inside the container
   python src/converters/dl_audio_to_midi_converter.py
   ```

4. When finished, you can exit the container and stop it:
   ```bash
   # Exit the container shell
   exit
   
   # Stop the container
   docker-compose down
   ```

## Integration with ASP.NET Core (Future Steps)

In the future, to integrate this with ASP.NET Core:

1. The ASP.NET Core application can call your Python scripts using `Process.Start()` to invoke Python scripts
2. Shared directories like `data` can be used to exchange files between systems

## Troubleshooting

- If you encounter issues with dependencies, you can modify the Dockerfile to install additional packages
- For file permission issues, you may need to adjust permissions in the container or on your host 