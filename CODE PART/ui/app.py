import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import time
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="✏️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained MNIST model"""
    try:
        model = tf.keras.models.load_model("models/mnist_cnn_model.keras")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.error("Please train the model first by running: `python train.py`")
        st.stop()

def preprocess_digit(image_data):
    """
    Simplified preprocessing to properly handle canvas data
    """
    if image_data is None:
        return np.zeros((1, 28, 28, 1)), np.zeros((28, 28))
    
    # Convert canvas data to grayscale - handle different formats
    if len(image_data.shape) == 3:
        if image_data.shape[2] == 4:  # RGBA
            # Use RGB channels, ignore alpha for now
            img = cv2.cvtColor(image_data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:  # RGB
            img = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img = image_data.astype(np.uint8)
    
    # Invert if needed (make drawing white on black background)
    if img.mean() > 127:
        img = 255 - img
    
    # Check if there's any content
    if np.sum(img) == 0:
        return np.zeros((1, 28, 28, 1)), np.zeros((28, 28))
    
    # Simple resize to 28x28
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Convert to float and normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized.reshape(1, 28, 28, 1), img_normalized

def show_sample_digits():
    """Show sample MNIST digits for reference"""
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), _ = mnist.load_data()
        
        # Show one example of each digit
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        axes = axes.ravel()
        
        for digit in range(10):
            idx = np.where(y_train == digit)[0][0]
            axes[digit].imshow(x_train[idx], cmap='gray')
            axes[digit].set_title(f'Digit {digit}', fontsize=12, fontweight='bold')
            axes[digit].axis('off')
        
        plt.suptitle('MNIST Sample Digits', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"Could not load sample digits: {str(e)}")

def show_confidence_meter(confidence):
    """Show a visual confidence meter"""
    if confidence > 0.8:
        color = "🟢"
        text = "High Confidence"
    elif confidence > 0.6:
        color = "🟡"
        text = "Medium Confidence"
    elif confidence > 0.4:
        color = "🟠"
        text = "Low Confidence"
    else:
        color = "🔴"
        text = "Very Low Confidence"
    
    st.write(f"{color} **{text}** ({confidence:.1%})")

def show_digit_challenge():
    """Fun challenge mode - random digit to draw"""
    if 'challenge_digit' not in st.session_state:
        st.session_state.challenge_digit = np.random.randint(0, 10)
        st.session_state.challenge_score = 0
        st.session_state.challenge_attempts = 0
    
    st.markdown("### 🎯 Challenge Mode")
    st.markdown(f"**Draw this digit: {st.session_state.challenge_digit}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score", st.session_state.challenge_score)
    with col2:
        st.metric("Attempts", st.session_state.challenge_attempts)
    
    return st.session_state.challenge_digit

def update_challenge_score(predicted, target, confidence):
    """Update challenge score"""
    st.session_state.challenge_attempts += 1
    if predicted == target:
        points = int(confidence * 10)  # Max 10 points for perfect confidence
        st.session_state.challenge_score += points
        st.balloons()
        st.success(f"🎉 Correct! +{points} points")
        # Generate new challenge
        st.session_state.challenge_digit = np.random.randint(0, 10)
    else:
        st.warning(f"❌ Try again! You drew {predicted}, but need {target}")

# Main App
def main():
    st.title("✏️ MNIST Digit Recognizer")
    st.markdown("Draw a digit (0-9) and let the AI recognize it!")
    
    # Add mode selection
    mode = st.selectbox("Choose mode:", ["🎨 Free Draw", "🎯 Challenge Mode"])
    
    # Add a note about model performance
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("🎯 This model achieves 99%+ accuracy on standard MNIST test data")
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        **Step-by-step:**
        1. Draw a digit in the canvas on the left
        2. Make sure it's **large and thick**
        3. Click "🔍 Predict Digit" 
        4. View the results on the right
        
        **Tips for better recognition:**
        - Draw digits **large** (use most of the canvas)
        - Make strokes **thick and bold**
        - Keep digits **centered**
        - Write **clearly** like printed numbers
        - Don't leave gaps in the digit
        """)
        
        st.header("🎯 Digit Guidelines")
        with st.expander("See digit examples"):
            st.markdown("""
            - **0**: Complete oval, no gaps
            - **1**: Straight vertical line
            - **2**: Clear curves and lines
            - **3**: Two clear curves
            - **4**: Connected horizontal and vertical lines
            - **6** & **9**: Clear curved shapes
            - **8**: Two connected loops
            """)
        
        st.header("📊 Model Info")
        if st.button("📖 Show MNIST Samples"):
            show_sample_digits()
        
        # Add drawing style options
        st.header("🎨 Drawing Settings")
        stroke_width = st.slider("Brush Size", 15, 45, 30)
        stroke_color = st.color_picker("Brush Color", "#FFFFFF")
        bg_color = st.color_picker("Background", "#000000")
        
        st.markdown("---")
        st.caption("Built with Streamlit & TensorFlow")
    
    # Initialize session state
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Challenge mode setup
    target_digit = None
    if mode == "🎯 Challenge Mode":
        target_digit = show_digit_challenge()
    
    # Main content area
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("🎨 Draw Your Digit Here")
        
        # Canvas for drawing
        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=400,
            width=400,
            drawing_mode="freedraw",
            point_display_radius=0,
            key=f"canvas_{st.session_state.canvas_key}"
        )
        
        # Canvas controls
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🗑️ Clear Canvas", type="secondary", use_container_width=True):
                st.session_state.canvas_key += 1
                st.rerun()
        
        with col_b:
            if st.button("🔄 Reset History", type="secondary", use_container_width=True):
                st.session_state.prediction_history = []
                if mode == "🎯 Challenge Mode":
                    st.session_state.challenge_score = 0
                    st.session_state.challenge_attempts = 0
                st.rerun()
        
        with col_c:
            # Auto-predict toggle
            auto_predict = st.checkbox("🤖 Auto-predict", help="Automatically predict when you draw")
    
    with col2:
        st.subheader("🤖 AI Recognition")
        
        if canvas.image_data is not None:
            # Process the image
            digit_tensor, digit_display = preprocess_digit(canvas.image_data)
            
            # Show preprocessed image
            st.write("**Preprocessed (28×28):**")
            if np.sum(digit_display) > 0:  # Only show if there's content
                fig, ax = plt.subplots(1, 1, figsize=(3, 3))
                ax.imshow(digit_display, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
                ax.set_title("Processed Image", fontsize=10)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            else:
                # Show placeholder when no drawing
                st.info("👆 Draw something above to see the preprocessed image")
            
            # Auto-predict or manual predict
            should_predict = auto_predict and np.sum(digit_tensor) > 0.05
            manual_predict = st.button("🔍 Predict Digit", type="primary", use_container_width=True)
            
            if should_predict or manual_predict:
                if np.sum(digit_tensor) > 0:  # Check if there's content
                    # Load model and make prediction
                    model = load_model()
                    
                    with st.spinner("🧠 Analyzing your drawing..."):
                        pred = model.predict(digit_tensor, verbose=0)[0]
                        predicted = np.argmax(pred)
                        confidence = pred[predicted]
                    
                    # Display main prediction
                    st.markdown("### 🎯 Prediction Results")
                    
                    # Large prediction display
                    st.markdown(f"## **Predicted Digit: {predicted}**")
                    
                    # Confidence meter
                    show_confidence_meter(confidence)
                    
                    # Challenge mode scoring
                    if mode == "🎯 Challenge Mode" and target_digit is not None:
                        update_challenge_score(predicted, target_digit, confidence)
                    
                    # Store prediction in history (avoid duplicates)
                    if not st.session_state.prediction_history or \
                       st.session_state.prediction_history[-1]['digit'] != predicted:
                        st.session_state.prediction_history.append({
                            'digit': predicted,
                            'confidence': confidence,
                            'image': digit_display.copy(),
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                    
                    # Show top 3 predictions
                    st.markdown("**🏆 Top 3 Predictions:**")
                    top_3_idx = np.argsort(pred)[-3:][::-1]
                    
                    for i, idx in enumerate(top_3_idx):
                        emoji = ["🥇", "🥈", "🥉"][i]
                        percentage = pred[idx] * 100
                        st.write(f"{emoji} **Digit {idx}**: {percentage:.1f}%")
                    
                    # Detailed probabilities in expander
                    with st.expander("📊 All Probabilities"):
                        # Create a horizontal bar chart
                        prob_data = {str(i): float(pred[i]) for i in range(10)}
                        st.bar_chart(prob_data)
                        
                        # Probability table
                        st.markdown("**Detailed Breakdown:**")
                        for i in range(10):
                            st.write(f"**{i}**: {pred[i]:.4f} ({pred[i]*100:.2f}%)")
                
                else:
                    st.warning("🚫 No drawing detected! Please draw a digit first.")
                    st.info("💡 Make sure to draw thick, bold lines that fill the canvas.")
    
    # Prediction History
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📈 Recent Predictions")
        
        # Show recent predictions (last 8)
        recent_predictions = st.session_state.prediction_history[-8:]
        
        # Create columns for history display
        num_cols = min(len(recent_predictions), 4)
        cols = st.columns(num_cols)
        
        for i, pred_data in enumerate(recent_predictions[-num_cols:]):
            with cols[i]:
                st.markdown(f"**Digit {pred_data['digit']}**")
                st.markdown(f"*{pred_data['confidence']:.1%} confidence*")
                st.caption(f"At {pred_data['timestamp']}")
                
                # Show small image
                if np.sum(pred_data['image']) > 0:
                    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
                    ax.imshow(pred_data['image'], cmap='gray', interpolation='nearest', vmin=0, vmax=1)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.write("Empty image")
        
        # Summary statistics
        if len(st.session_state.prediction_history) > 1:
            st.markdown("**📊 Session Stats:**")
            predictions = [p['digit'] for p in st.session_state.prediction_history]
            confidences = [p['confidence'] for p in st.session_state.prediction_history]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(predictions))
            with col2:
                st.metric("Avg Confidence", f"{np.mean(confidences):.1%}")
            with col3:
                most_common = max(set(predictions), key=predictions.count)
                st.metric("Most Drawn", f"Digit {most_common}")
            with col4:
                high_conf = sum(1 for c in confidences if c > 0.8)
                st.metric("High Confidence", f"{high_conf}/{len(confidences)}")
    
    # Debug section (collapsible)
    with st.expander("🔧 Debug Information", expanded=False):
        if canvas.image_data is not None:
            digit_tensor, digit_display = preprocess_digit(canvas.image_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Canvas Data:**")
                st.write(f"Shape: {canvas.image_data.shape}")
                if len(canvas.image_data.shape) == 3 and canvas.image_data.shape[2] == 4:
                    alpha = canvas.image_data[:, :, 3]
                    st.write(f"Alpha range: {alpha.min()} to {alpha.max()}")
                    st.write(f"Alpha non-zero pixels: {np.count_nonzero(alpha)}")
            
            with col2:
                st.write("**Processed Data:**")
                st.write(f"Tensor shape: {digit_tensor.shape}")
                st.write(f"Value range: {digit_tensor.min():.3f} to {digit_tensor.max():.3f}")
                st.write(f"Non-zero pixels: {np.count_nonzero(digit_tensor)}")
                st.write(f"Pixel sum: {np.sum(digit_tensor):.3f}")
            
            # Show processing steps
            if len(canvas.image_data.shape) == 3:
                st.write("**Processing Steps:**")
                
                # Show different processing stages
                if canvas.image_data.shape[2] == 4:  # RGBA
                    rgb_img = canvas.image_data[:, :, :3]
                    gray_img = cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:  # RGB
                    gray_img = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                
                if np.sum(gray_img) > 0:  # Only show if there's content
                    # Invert if needed
                    if gray_img.mean() > 127:
                        processed_img = 255 - gray_img
                    else:
                        processed_img = gray_img
                    
                    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                    
                    # Show original grayscale
                    axes[0].imshow(gray_img, cmap='gray', vmin=0, vmax=255)
                    axes[0].set_title("Original Grayscale")
                    axes[0].axis('off')
                    
                    # Show after inversion
                    axes[1].imshow(processed_img, cmap='gray', vmin=0, vmax=255)
                    axes[1].set_title("After Inversion")
                    axes[1].axis('off')
                    
                    # Show final 28x28
                    final_28 = cv2.resize(processed_img, (28, 28), interpolation=cv2.INTER_AREA)
                    axes[2].imshow(final_28, cmap='gray', vmin=0, vmax=255)
                    axes[2].set_title("Final 28x28")
                    axes[2].axis('off')
                    
                    st.pyplot(fig)
                    plt.close()

if __name__ == "__main__":
    main()