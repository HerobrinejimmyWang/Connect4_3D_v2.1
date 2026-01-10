import sys
import multiprocessing
from trainer import Trainer, TrainerArgs

# Protect entry point for multiprocessing
if __name__ == "__main__":
    # Ensure standard python multiprocessing behavior
    multiprocessing.freeze_support()
    
    # Define Args
    args = TrainerArgs()
    
    # --- Custom Configuration ---
    args.num_iterations = 40      # Adjust based on how long you want to run
    args.num_self_play_games = 100 # Increase this if using parallel to get better data
    args.checkpoint_interval = 5
    args.epochs = 5
    
    # --- Resume Training? ---
    # To resume, provide path: e.g., './checkpoints/checkpoint_10.pth.tar'
    # To start new, use None
    resume_checkpoint = None 
    # resume_checkpoint = './checkpoints/latest.pth.tar' 
    
    print("Initializing Training...")
    trainer = Trainer(args, resume_path=resume_checkpoint)
    
    print("Starting Training Loop...")
    trainer.train()