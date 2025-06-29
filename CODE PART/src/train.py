import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_and_preprocess_data():
    """Load and preprocess MNIST data"""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print("Data preprocessing completed!")
    return (x_train, y_train), (x_test, y_test)

def create_cnn_model():
    """Create CNN model architecture"""
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def create_simple_model():
    """Create a simpler, faster model for quick training"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, x_test, y_test):
    """Evaluate model performance"""
    print("\n" + "="*50)
    print("EVALUATING MODEL PERFORMANCE")
    print("="*50)
    
    # Make predictions
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=[str(i) for i in range(10)]))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=list(range(10)))
    
    # Show some predictions
    show_sample_predictions(model, x_test, y_test, num_samples=10)
    
    return test_accuracy

def show_sample_predictions(model, x_test, y_test, num_samples=10):
    """Show sample predictions"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    # Get random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Make prediction
        pred_proba = model.predict(x_test[idx:idx+1], verbose=0)[0]
        pred_class = np.argmax(pred_proba)
        true_class = np.argmax(y_test[idx])
        confidence = pred_proba[pred_class]
        
        # Plot image
        axes[i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'True: {true_class}, Pred: {pred_class}\nConf: {confidence:.2f}', 
                         fontsize=10)
        axes[i].axis('off')
        
        # Color the title based on correctness
        if pred_class == true_class:
            axes[i].title.set_color('green')
        else:
            axes[i].title.set_color('red')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=14)
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("="*60)
    print("MNIST DIGIT RECOGNITION - MODEL TRAINING")
    print("="*60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Ask user for model complexity
    print("\nChoose model complexity:")
    print("1. Simple model (faster training, ~98% accuracy)")
    print("2. Complex model (slower training, ~99%+ accuracy)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\nCreating simple CNN model...")
        model = create_simple_model()
        epochs = 10
        batch_size = 128
    else:
        print("\nCreating complex CNN model...")
        model = create_cnn_model()
        epochs = 20
        batch_size = 64
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    print("="*50)
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nTraining completed!")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    final_accuracy = evaluate_model(model, x_test, y_test)
    
    # Save final model
    model_path = 'models/mnist_cnn_model.keras'
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save model info
    with open('models/model_info.txt', 'w') as f:
        f.write(f"MNIST CNN Model Information\n")
        f.write(f"="*30 + "\n")
        f.write(f"Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Training Epochs: {len(history.history['accuracy'])}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Model Type: {'Simple' if choice == '1' else 'Complex'}\n")
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"✅ Model trained successfully!")
    print(f"📊 Final accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"💾 Model saved to: {model_path}")
    print(f"📈 Training plots saved as PNG files")
    print(f"🚀 Ready to use with Streamlit app!")
    print("="*60)
    
    # Test model loading
    print("\nTesting model loading...")
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        test_pred = loaded_model.predict(x_test[:1], verbose=0)
        print("✅ Model loading test successful!")
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")

if __name__ == "__main__":
    main()