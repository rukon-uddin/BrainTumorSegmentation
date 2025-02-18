from argparse import ArgumentParser

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
import os
from datetime import datetime

from keras.layers import ELU

from utils.dataset import MRIDataset
from utils.loss import dice_loss_binary
from utils.metric import dice_coef
from utils.models import binary_model
from utils.optimizers import LH_Adam


def train(dataset_dir: str, binary_weights_path: str):
    log_dir = os.path.join('/content/drive/MyDrive/Fiver_Projects/allu/logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    
    # Create CSV log path
    csv_log_path = os.path.join(log_dir, 'training_log.csv')


    dataset = MRIDataset(binary_dataset_path=dataset_dir)

    batch_size = 1
    train_img_datagen = dataset.binary_train_datagen(batch_size)
    val_img_datagen = dataset.binary_val_datagen(batch_size)

    steps_per_epoch, val_steps_per_epoch = dataset.binary_steps_per_epoch(batch_size)

    n_channels = 20
    model = binary_model(128, 128, 128, 4, 1, n_channels, activation=ELU())

    learning_rate = 0.0003
    optimizer = LH_Adam(learning_rate)

    model.compile(optimizer=optimizer, loss=dice_loss_binary, metrics=[dice_coef])

    # checkpoint_callback = ModelCheckpoint(filepath=binary_weights_path, save_weights_only=True, save_best_only=True)
    callbacks = [
        ModelCheckpoint(filepath=binary_weights_path, save_weights_only=True), 
        
        CSVLogger(
            csv_log_path,
            append=True  
        ),

        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0
        )]

    model.fit(x=train_img_datagen,
              steps_per_epoch=steps_per_epoch,
              epochs=300,
              verbose=1,
              validation_data=val_img_datagen,
              validation_steps=val_steps_per_epoch,
              callbacks=callbacks,)
    



def resumeTrain(dataset_dir: str, binary_weights_path: str, initial_epoch: int = 0):
    # Create logs directory
    log_dir = os.path.join('/content/drive/MyDrive/Fiver_Projects/allu/logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    
    # Create CSV log path
    csv_log_path = os.path.join(log_dir, 'training_log.csv')
    
    dataset = MRIDataset(binary_dataset_path=dataset_dir)
    batch_size = 1
    train_img_datagen = dataset.binary_train_datagen(batch_size)
    val_img_datagen = dataset.binary_val_datagen(batch_size)
    steps_per_epoch, val_steps_per_epoch = dataset.binary_steps_per_epoch(batch_size)
    
    n_channels = 20
    model = binary_model(128, 128, 128, 4, 1, n_channels, activation=ELU())
    learning_rate = 0.0003
    optimizer = LH_Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=dice_loss_binary, metrics=[dice_coef])
    
    # Load previous weights if they exist
    if os.path.exists(binary_weights_path):
        print(f"Loading weights from {binary_weights_path}")
        model.load_weights(binary_weights_path)
    
    # Define callbacks
    callbacks = [
        # Model checkpoint
        ModelCheckpoint(
            filepath=binary_weights_path,
            save_weights_only=True
        ),
        # CSV Logger
        CSVLogger(
            csv_log_path,
            append=True  # Append if file exists
        ),
        # TensorBoard callback
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0
        )
    ]
    
    # Train model
    history = model.fit(
        x=train_img_datagen,
        steps_per_epoch=steps_per_epoch,
        epochs=300,
        verbose=1,
        validation_data=val_img_datagen,
        validation_steps=val_steps_per_epoch,
        callbacks=callbacks,
        initial_epoch=initial_epoch
    )
    
    return history, log_dir







if __name__ == '__main__':
    arg_parser = ArgumentParser()

    arg_parser.add_argument("-d", "--dataset_dir",
                            help="Directory for the training dataset. Should contain the 'train' and 'val' "
                                 "directories.",
                            required=True)
    arg_parser.add_argument("-w", "--binary_weights",
                            help="Path to the binary model's weights. This is the path of where to save the weights",
                            required=True)

    args = arg_parser.parse_args()

    # Check Weights File
    if not args.binary_weights.endswith(".hdf5"):
        raise ValueError("Invalid weight file format")

    train(args.dataset_dir, args.binary_weights)
    


# if __name__ == '__main__':
#     arg_parser = ArgumentParser()
#     arg_parser.add_argument(
#         "-d", "--dataset_dir",
#         help="Directory for the training dataset. Should contain the 'train' and 'val' directories.",
#         required=True
#     )
#     arg_parser.add_argument(
#         "-w", "--binary_weights",
#         help="Path to the binary model's weights. This is the path of where to save the weights",
#         required=True
#     )
#     arg_parser.add_argument(
#         "-e", "--initial_epoch",
#         help="Epoch to resume training from",
#         type=int,
#         default=0
#     )
#     args = arg_parser.parse_args()

#     # Check Weights File
#     if not args.binary_weights.endswith(".hdf5"):
#         raise ValueError("Invalid weight file format")
        
#     history, log_dir = train(args.dataset_dir, args.binary_weights, args.initial_epoch)
#     print(f"Training logs saved to: {log_dir}")