from ultralytics import YOLO
#####################################################
# Define a function to plot learning curves for loss values
def plot_learning_curve(df, train_loss_col, val_loss_col, title):
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x='epoch', y=train_loss_col, label='Train Loss', color='#141140', linestyle='-', linewidth=2)
    sns.lineplot(data=df, x='epoch', y=val_loss_col, label='Validation Loss', color='orangered', linestyle='--', linewidth=2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
#########################################################
# Initialize YOLOv8 model
# Available options: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x' for varying sizes
model = YOLO('yolov8n.pt')  # Load pretrained weights for YOLOv8 nano (lightweight model)
results = model.train(
    data='./top-view-vehicle-detection-image-dataset/Vehicle_Detection_Image_Dataset/data.yaml',
    epochs=100,              # Number of epochs to train for
    imgsz=640,               # Size of input images as integer
    device='cpu',                # Device to run on, i.e. cuda device=0 
    patience=50,             # Epochs to wait for no observable improvement for early stopping of training
    batch=32,                # Number of images per batch
    optimizer='auto',        # Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    lr0=0.0001,              # Initial learning rate 
    lrf=0.1,                 # Final learning rate (lr0 * lrf)
    dropout=0.1,             # Use dropout regularization
    seed=0         
)
# Save the trained model
model_path = 'Model01.pt'
model.save(model_path)
print(f"Model saved to {model_path}")
####################################################
