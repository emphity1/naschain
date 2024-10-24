import os
import torch
import asyncio
import bittensor as bt

# Import ValiTrainer and load_model from vali_trainer.py
from vali_trainer import ValiTrainer  # Ensure vali_trainer.py defines these


def load_model(model_dir):
    try:
        model = torch.jit.load(model_dir)
        bt.logging.info("Torch script model loaded using torch.jit.load")
        return model
    except Exception as e:
        bt.logging.warning(f"torch.jit.load failed with error: {e}")
        try:
            model = torch.load(model_dir)
            bt.logging.info("Model loaded using torch.load")
            return model
        except Exception as jit_e:
            bt.logging.error(f"torch.load also failed with error: {jit_e}")
            raise  #


class Validator:
    def __init__(self):
        self.should_exit = False

    async def forward(self):
        trainer = ValiTrainer(epochs=50)

        # Path to the directory containing your model files
        model_dir = "/workspace/Dima/models/"

        # Ensure the model directory exists
        if not os.path.exists(model_dir):
            print(f"Model directory {model_dir} does not exist.")
            return

        log_file_path = os.path.join(model_dir, "retrain_results_log.txt")
        # Open the log file in append mode
        with open(log_file_path, "a") as log_file:
            # Loop through all files in the directory
            for model_file in os.listdir(model_dir):
                if model_file.endswith(".pt"):  # Process only .pt files
                    try:
                        model_path = os.path.join(model_dir, model_file)

                        # Start with initial learning rate 0.06 and add 0.005 every iteration
                        lr = 0.06
                        while lr <= 2.0:
                            print(f"Processing model: {model_path}")
                            # Load the model
                            model = load_model(model_path)
                            model.to("cuda")  # Move model to GPU if needed
                            print(f"Retraining model with learning rate: {lr}")
                            
                            # Initialize the trainer with the current learning rate
                            trainer = ValiTrainer(epochs=50, learning_rate=lr)
                            trainer.initialize_weights(model)
                            retrained_model = trainer.train(model)
                            
                            # Test the retrained model
                            new_accuracy = trainer.test(retrained_model)
                            
                            # Save the retrained model with learning rate info in the filename
                            lr_formatted = f"{lr:.3f}".replace('.', '_')
                            lr_model_path = model_path.replace(".pt", f"_lr{lr_formatted}.pt")
                            scripted_model = torch.jit.script(retrained_model)
                            scripted_model.save(lr_model_path)
                            print(f"Retrained model saved at: {lr_model_path}")

                            # Log the result with learning rate
                            log_file.write(f"Model File: {model_file}\n")
                            log_file.write(f"Learning Rate: {lr}\n")
                            log_file.write(f"New Accuracy: {new_accuracy}\n")
                            log_file.write("-" * 40 + "\n")
                            
                            # Flush the buffer after each iteration to ensure logs are written to disk
                            log_file.flush()
                            
                            # Increment learning rate by 0.005
                            lr += 0.005
                            lr = round(lr, 3)  # To prevent floating point precision issues
                        
                    except Exception as e:
                        # Handle any exceptions and log the error
                        log_file.write(f"Error processing {model_file}: {str(e)}\n")
                        log_file.write("-" * 40 + "\n")
                        log_file.flush()
                        print(f"Error processing {model_file}: {str(e)}")

        # After processing all models, you can set a flag to exit the loop
        self.should_exit = True

    async def run(self):
        await self.forward()

if __name__ == "__main__":
    validator = Validator()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(validator.run())
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting.")
    finally:
        loop.close()
