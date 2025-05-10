#!/usr/bin/env python
import os
import sys
import subprocess
import time
from script_running_constants import (
    RUN_CNN_CQT_DEFAULT,
    RUN_CNN_MEL_DEFAULT
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

def run_cnn_cqt():
    """Run CNN model with CQT features"""
    print("\nRunning CNN model with CQT features...\n")
    return run_command(RUN_CNN_CQT_DEFAULT)

def run_cnn_mel():
    """Run CNN model with MEL features"""
    print("\nRunning CNN model with MEL features...\n")
    return run_command(RUN_CNN_MEL_DEFAULT)

def run_cnn_both():
    """Run both CNN models sequentially"""
    print("\nRunning both CNN models...\n")
    print("\n--- First: CNN with CQT features ---\n")
    run_cnn_cqt()
    print("\n--- Second: CNN with MEL features ---\n")
    run_cnn_mel()
    return 0

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_menu():
    """Display the main menu."""
    clear_screen()
    print("\n" + "=" * 50)
    print("             MODEL TRAINING SCRIPT RUNNER")
    print("=" * 50)
    print("\nAvailable options:")
    print("  1. Run CNN model with CQT features")
    print("  2. Run CNN model with MEL features")
    print("  3. Run both CNN models (CQT then MEL)")
    print("  4. Show available commands")
    print("  5. Exit")
    print("\n" + "-" * 50)
    return input("Enter your choice (1-5): ")

def show_commands():
    """Show the available commands."""
    clear_screen()
    print("\nAvailable commands:")
    print(f"  1. CNN with CQT features: {RUN_CNN_CQT_DEFAULT}")
    print(f"  2. CNN with MEL features: {RUN_CNN_MEL_DEFAULT}")
    print("\nPress Enter to return to menu...")
    input()

def interactive_menu():
    """Run an interactive console menu."""
    while True:
        choice = display_menu()
        
        if choice == "1":
            run_cnn_cqt()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "2":
            run_cnn_mel()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "3":
            run_cnn_both()
            print("\nBoth commands execution finished. Press Enter to return to menu...")
            input()
        elif choice == "4":
            show_commands()
        elif choice == "5":
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
