🎧 AI-Powered Signal Denoising System

An interactive Streamlit-based web application for denoising signals across multiple domains including speech, wireless, and ECG signals. The system combines deep learning (autoencoder model) with classical filtering techniques to provide efficient and accurate noise removal.

🚀 Features
🔊 Speech (audio) denoising with playback support
📡 Wireless signal denoising
🫀 ECG signal denoising
🤖 AI-based denoising using a trained deep learning model
⚙️ Classical filters: Wiener Filter & Median Filter
📊 Performance metrics: SNR, MSE, RMSE
📈 Advanced visualizations (time-domain, frequency, spectrograms)
🎙️ Real-time microphone denoising (beta)
💾 Download processed signals
🛠️ Technologies Used
Python
Streamlit
TensorFlow / Keras
NumPy, SciPy
Librosa
Matplotlib
Pandas
📂 Project Structure
├── app_final_enhanced.py   # Main Streamlit application
├── data/
│   └── denoising_model.h5  # Trained AI model
├── train_model.py          # Model architecture (if used)
⚙️ Installation
Clone the repository:
git clone <your-repo-link>
cd <repo-name>
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app_final_enhanced.py
📌 How It Works
Upload a noisy signal (.wav, .npy, .csv, .txt) or record audio
Choose a denoising method (AI / Wiener / Median)
View denoised output with visual comparisons
Analyze performance using metrics and plots
Download the processed signal
🎯 Applications
Speech enhancement
Biomedical signal processing (ECG)
Wireless communication noise reduction
Signal analysis and research
📊 Output
Cleaned (denoised) signals
Visual comparison graphs
Audio playback (for speech)
Performance evaluation metrics
✨ Future Improvements
Real-time AI denoising optimization
Support for more signal types
Improved model accuracy
Deployment on cloud platforms
