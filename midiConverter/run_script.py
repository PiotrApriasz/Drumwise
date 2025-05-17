#!/usr/bin/env python
import os
import sys
import subprocess
import time
from script_running_constants import (
    # Training commands
    RUN_CNN_CQT_DEFAULT,
    RUN_CNN_MEL_DEFAULT,
    RUN_LSTM_CQT_DEFAULT,
    RUN_LSTM_MEL_DEFAULT,
    RUN_TRANSFORMER_CQT_DEFAULT,
    RUN_TRANSFORMER_MEL_DEFAULT,
    RUN_CNN_LSTM_CQT_DEFAULT,
    RUN_CNN_LSTM_MEL_DEFAULT,
    # Evaluation commands
    EVAL_CNN_CQT_DEFAULT,
    EVAL_CNN_MEL_DEFAULT,
    EVAL_LSTM_CQT_DEFAULT,
    EVAL_LSTM_MEL_DEFAULT,
    EVAL_TRANSFORMER_CQT_DEFAULT,
    EVAL_TRANSFORMER_MEL_DEFAULT,
    EVAL_CNN_LSTM_CQT_DEFAULT,
    EVAL_CNN_LSTM_MEL_DEFAULT,
    EVAL_ENSEMBLE_DEFAULT
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

def run_lstm_cqt():
    """Run LSTM model with CQT features"""
    print("\nRunning LSTM model with CQT features...\n")
    return run_command(RUN_LSTM_CQT_DEFAULT)

def run_lstm_mel():
    """Run LSTM model with MEL features"""
    print("\nRunning LSTM model with MEL features...\n")
    return run_command(RUN_LSTM_MEL_DEFAULT)

def run_transformer_cqt():
    """Run Transformer model with CQT features"""
    print("\nRunning Transformer model with CQT features...\n")
    return run_command(RUN_TRANSFORMER_CQT_DEFAULT)

def run_transformer_mel():
    """Run Transformer model with MEL features"""
    print("\nRunning Transformer model with MEL features...\n")
    return run_command(RUN_TRANSFORMER_MEL_DEFAULT)

def run_cnn_lstm_cqt():
    """Run CNN-LSTM model with CQT features"""
    print("\nRunning CNN-LSTM model with CQT features...\n")
    return run_command(RUN_CNN_LSTM_CQT_DEFAULT)

def run_cnn_lstm_mel():
    """Run CNN-LSTM model with MEL features"""
    print("\nRunning CNN-LSTM model with MEL features...\n")
    return run_command(RUN_CNN_LSTM_MEL_DEFAULT)

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

def eval_lstm_cqt():
    """Evaluate LSTM model with CQT features"""
    print("\nEvaluating LSTM model with CQT features...\n")
    return run_command(EVAL_LSTM_CQT_DEFAULT)

def eval_lstm_mel():
    """Evaluate LSTM model with MEL features"""
    print("\nEvaluating LSTM model with MEL features...\n")
    return run_command(EVAL_LSTM_MEL_DEFAULT)

def eval_transformer_cqt():
    """Evaluate Transformer model with CQT features"""
    print("\nEvaluating Transformer model with CQT features...\n")
    return run_command(EVAL_TRANSFORMER_CQT_DEFAULT)

def eval_transformer_mel():
    """Evaluate Transformer model with MEL features"""
    print("\nEvaluating Transformer model with MEL features...\n")
    return run_command(EVAL_TRANSFORMER_MEL_DEFAULT)

def eval_cnn_lstm_cqt():
    """Evaluate CNN-LSTM model with CQT features"""
    print("\nEvaluating CNN-LSTM model with CQT features...\n")
    return run_command(EVAL_CNN_LSTM_CQT_DEFAULT)

def eval_cnn_lstm_mel():
    """Evaluate CNN-LSTM model with MEL features"""
    print("\nEvaluating CNN-LSTM model with MEL features...\n")
    return run_command(EVAL_CNN_LSTM_MEL_DEFAULT)

def eval_ensemble():
    """Evaluate CNN ensemble (MEL+CQT)"""
    print("\nEvaluating CNN ensemble model (MEL+CQT)...\n")
    return run_command(EVAL_ENSEMBLE_DEFAULT)

def eval_all():
    """Evaluate all models sequentially"""
    print("\nEvaluating all models sequentially...\n")

    print("\n--- Evaluating CNN with CQT features ---\n")
    eval_cnn_cqt()

    print("\n--- Evaluating CNN with MEL features ---\n")
    eval_cnn_mel()

    print("\n--- Evaluating LSTM with CQT features ---\n")
    eval_lstm_cqt()

    print("\n--- Evaluating LSTM with MEL features ---\n")
    eval_lstm_mel()

    print("\n--- Evaluating Transformer with CQT features ---\n")
    eval_transformer_cqt()

    print("\n--- Evaluating Transformer with MEL features ---\n")
    eval_transformer_mel()

    print("\n--- Evaluating CNN-LSTM with CQT features ---\n")
    eval_cnn_lstm_cqt()

    print("\n--- Evaluating CNN-LSTM with MEL features ---\n")
    eval_cnn_lstm_mel()

    print("\n--- Evaluating CNN ensemble (MEL+CQT) ---\n")
    eval_ensemble()

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
    print("  4. Train LSTM model with CQT features")
    print("  5. Train LSTM model with MEL features")
    print("  6. Train Transformer model with CQT features")
    print("  7. Train Transformer model with MEL features")
    print("  8. Train CNN-LSTM model with CQT features")
    print("  9. Train CNN-LSTM model with MEL features")
    print("  0. Back to main menu")
    print("\n" + "-" * 50)
    return input("Enter your choice (0-9): ")

def display_evaluation_menu():
    """Display the evaluation menu."""
    clear_screen()
    print("\n" + "=" * 50)
    print("              MODEL EVALUATION OPTIONS")
    print("=" * 50)
    print("\nAvailable evaluation options:")
    print("  1. Evaluate CNN model with CQT features")
    print("  2. Evaluate CNN model with MEL features")
    print("  3. Evaluate LSTM model with CQT features")
    print("  4. Evaluate LSTM model with MEL features")
    print("  5. Evaluate Transformer model with CQT features")
    print("  6. Evaluate Transformer model with MEL features")
    print("  7. Evaluate CNN-LSTM model with CQT features")
    print("  8. Evaluate CNN-LSTM model with MEL features")
    print("  9. Evaluate CNN ensemble (MEL+CQT)")
    print("  A. Evaluate all models")
    print("  0. Back to main menu")
    print("\n" + "-" * 50)
    return input("Enter your choice (0-9, A): ")

def show_commands():
    """Show the available commands."""
    clear_screen()
    print("\n" + "=" * 50)
    print("              AVAILABLE COMMANDS")
    print("=" * 50)

    print("\nTraining commands:")
    print(f"  1. CNN with CQT features: {RUN_CNN_CQT_DEFAULT}")
    print(f"  2. CNN with MEL features: {RUN_CNN_MEL_DEFAULT}")
    print(f"  3. LSTM with CQT features: {RUN_LSTM_CQT_DEFAULT}")
    print(f"  4. LSTM with MEL features: {RUN_LSTM_MEL_DEFAULT}")
    print(f"  5. Transformer with CQT features: {RUN_TRANSFORMER_CQT_DEFAULT}")
    print(f"  6. Transformer with MEL features: {RUN_TRANSFORMER_MEL_DEFAULT}")
    print(f"  7. CNN-LSTM with CQT features: {RUN_CNN_LSTM_CQT_DEFAULT}")
    print(f"  8. CNN-LSTM with MEL features: {RUN_CNN_LSTM_MEL_DEFAULT}")

    print("\nEvaluation commands:")
    print(f"  9. CNN with CQT features: {EVAL_CNN_CQT_DEFAULT}")
    print(f" 10. CNN with MEL features: {EVAL_CNN_MEL_DEFAULT}")
    print(f" 11. LSTM with CQT features: {EVAL_LSTM_CQT_DEFAULT}")
    print(f" 12. LSTM with MEL features: {EVAL_LSTM_MEL_DEFAULT}")
    print(f" 13. Transformer with CQT features: {EVAL_TRANSFORMER_CQT_DEFAULT}")
    print(f" 14. Transformer with MEL features: {EVAL_TRANSFORMER_MEL_DEFAULT}")
    print(f" 15. CNN-LSTM with CQT features: {EVAL_CNN_LSTM_CQT_DEFAULT}")
    print(f" 16. CNN-LSTM with MEL features: {EVAL_CNN_LSTM_MEL_DEFAULT}")
    print(f" 17. CNN ensemble: {EVAL_ENSEMBLE_DEFAULT}")

    print("\nPress Enter to return to menu...")
    input()

def handle_training_menu():
    """Handle the training menu options."""
    while True:
        choice = display_training_menu()
        
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
            run_lstm_cqt()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "5":
            run_lstm_mel()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "6":
            run_transformer_cqt()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "7":
            run_transformer_mel()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "8":
            run_cnn_lstm_cqt()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "9":
            run_cnn_lstm_mel()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "0":
            return
        else:
            print("\nInvalid choice. Please try again.")
            time.sleep(1.5)

def handle_evaluation_menu():
    """Handle the evaluation menu options."""
    while True:
        choice = display_evaluation_menu()

        if choice == "1":
            eval_cnn_cqt()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "2":
            eval_cnn_mel()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "3":
            eval_lstm_cqt()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "4":
            eval_lstm_mel()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "5":
            eval_transformer_cqt()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "6":
            eval_transformer_mel()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "7":
            eval_cnn_lstm_cqt()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "8":
            eval_cnn_lstm_mel()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice == "9":
            eval_ensemble()
            print("\nCommand execution finished. Press Enter to return to menu...")
            input()
        elif choice.upper() == "A":
            eval_all()
            print("\nAll evaluations completed. Press Enter to return to menu...")
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
