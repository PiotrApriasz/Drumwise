#!/usr/bin/env python
import os
import sys
import subprocess
import time
from script_running_constants import (
    RUN_CNN_DEFAULT,
    EVAL_CNN_DEFAULT,
)

def run_command(command, verbose=True):
    """
    Run a shell command with optional verbosity.

    Args:
        command (str): The command to execute
        verbose (bool): Whether to print command before executing

    Returns:
        The return code from the command
    """
    if verbose:
        print(f"\nExecuting command: {command}\n")
        print("-" * 50)

    # Use shell=True since our commands are defined as shell commands
    return subprocess.run(command, shell=True).returncode

def run_cnn():
    """Train the CNN baseline (MEL features)"""
    print("\nTraining the CNN baseline...\n")
    return run_command(RUN_CNN_DEFAULT)

def eval_cnn():
    """Evaluate the CNN baseline"""
    print("\nEvaluating the CNN baseline...\n")
    return run_command(EVAL_CNN_DEFAULT)

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_main_menu():
    """Display the main menu."""
    clear_screen()
    print("\n" + "=" * 50)
    print("          MODEL TRAINING & EVALUATION RUNNER")
    print("=" * 50)
    print("\nMain menu:")
    print("  1. Train the CNN baseline")
    print("  2. Evaluate the CNN baseline")
    print("  3. Show all available commands")
    print("  4. Exit")
    print("\n" + "-" * 50)
    return input("Enter your choice (1-4): ")

def show_commands():
    """Show the available commands."""
    clear_screen()
    print("\n" + "=" * 50)
    print("              AVAILABLE COMMANDS")
    print("=" * 50)
    print(f"\nTraining:   {RUN_CNN_DEFAULT}")
    print(f"Evaluation: {EVAL_CNN_DEFAULT}")
    print("\nPress Enter to return to menu...")
    input()

def interactive_menu():
    """Run an interactive console menu."""
    actions = {"1": run_cnn, "2": eval_cnn}
    while True:
        choice = display_main_menu()

        if choice in actions:
            actions[choice]()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "3":
            show_commands()
        elif choice == "4":
            print("\nExiting program. Goodbye!")
            time.sleep(1)
            sys.exit(0)
        else:
            print("\nInvalid choice. Please try again.")
            time.sleep(1.5)

if __name__ == "__main__":
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting...")
        sys.exit(0)
