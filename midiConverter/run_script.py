#!/usr/bin/env python
import os
import sys
import subprocess
import time
from script_running_constants import (
    # Training commands
    RUN_CNN_CQT_DEFAULT,
    RUN_CNN_MEL_DEFAULT,
    # Evaluation commands
    EVAL_CNN_CQT_DEFAULT,
    EVAL_CNN_MEL_DEFAULT,
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

# Training functions
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

# Evaluation functions
def eval_cnn_cqt():
    """Evaluate CNN model with CQT features"""
    print("\nEvaluating CNN model with CQT features...\n")
    return run_command(EVAL_CNN_CQT_DEFAULT)

def eval_cnn_mel():
    """Evaluate CNN model with MEL features"""
    print("\nEvaluating CNN model with MEL features...\n")
    return run_command(EVAL_CNN_MEL_DEFAULT)

def eval_all():
    """Evaluate all models sequentially"""
    print("\nEvaluating all models sequentially...\n")

    print("\n--- Evaluating CNN with CQT features ---\n")
    eval_cnn_cqt()

    print("\n--- Evaluating CNN with MEL features ---\n")
    eval_cnn_mel()

    return 0

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_main_menu():
    """Display the main category menu."""
    clear_screen()
    print("\n" + "=" * 50)
    print("          MODEL TRAINING & EVALUATION RUNNER")
    print("=" * 50)
    print("\nMain menu:")
    print("  1. Training options")
    print("  2. Evaluation options")
    print("  3. Show all available commands")
    print("  4. Exit")
    print("\n" + "-" * 50)
    return input("Enter your choice (1-4): ")

def display_training_menu():
    """Display the training menu."""
    clear_screen()
    print("\n" + "=" * 50)
    print("               MODEL TRAINING OPTIONS")
    print("=" * 50)
    print("\nAvailable training options:")
    print("  1. Train CNN model with CQT features")
    print("  2. Train CNN model with MEL features")
    print("  3. Train both CNN models (CQT then MEL)")
    print("  0. Back to main menu")
    print("\n" + "-" * 50)
    return input("Enter your choice (0-3): ")

def display_evaluation_menu():
    """Display the evaluation menu."""
    clear_screen()
    print("\n" + "=" * 50)
    print("              MODEL EVALUATION OPTIONS")
    print("=" * 50)
    print("\nAvailable evaluation options:")
    print("  1. Evaluate CNN model with CQT features")
    print("  2. Evaluate CNN model with MEL features")
    print("  A. Evaluate all models")
    print("  0. Back to main menu")
    print("\n" + "-" * 50)
    return input("Enter your choice (0-2, A): ")

def show_commands():
    """Show the available commands."""
    clear_screen()
    print("\n" + "=" * 50)
    print("              AVAILABLE COMMANDS")
    print("=" * 50)

    print("\nTraining commands:")
    print(f"  1. CNN with CQT features: {RUN_CNN_CQT_DEFAULT}")
    print(f"  2. CNN with MEL features: {RUN_CNN_MEL_DEFAULT}")

    print("\nEvaluation commands:")
    print(f"  3. CNN with CQT features: {EVAL_CNN_CQT_DEFAULT}")
    print(f"  4. CNN with MEL features: {EVAL_CNN_MEL_DEFAULT}")

    print("\nPress Enter to return to menu...")
    input()

def handle_training_menu():
    """Handle the training menu options."""
    actions = {"1": run_cnn_cqt, "2": run_cnn_mel, "3": run_cnn_both}
    while True:
        choice = display_training_menu()

        if choice in actions:
            actions[choice]()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "0":
            return
        else:
            print("\nInvalid choice. Please try again.")
            time.sleep(1.5)

def handle_evaluation_menu():
    """Handle the evaluation menu options."""
    actions = {"1": eval_cnn_cqt, "2": eval_cnn_mel, "A": eval_all}
    while True:
        choice = display_evaluation_menu().upper()

        if choice in actions:
            actions[choice]()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "0":
            return
        else:
            print("\nInvalid choice. Please try again.")
            time.sleep(1.5)

def interactive_menu():
    """Run an interactive console menu."""
    while True:
        choice = display_main_menu()

        if choice == "1":
            handle_training_menu()
        elif choice == "2":
            handle_evaluation_menu()
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
