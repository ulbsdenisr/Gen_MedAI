# MedAI Chat Interface - Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python app.py
```

The server will start at `http://localhost:5000`

### 3. Open in Browser
- Open your web browser and navigate to: **http://localhost:5000**
- You should see the MedAI Chat interface

## 🎯 Features

- **💬 Natural Chat Interface**: Clean, intuitive chat design similar to popular messaging apps
- **🏥 Medical Symptom Analysis**: Automatically extracts medical symptoms from your input using NER
- **📚 Relevant Information**: Retrieves relevant medical information using RAG (Retrieval-Augmented Generation)
- **🎨 Beautiful UI**: Modern, responsive design that works on desktop and mobile
- **⚡ Real-time Processing**: Instant analysis and response generation

## 📋 How to Use

1. **Describe Your Symptoms**: Type your medical symptoms or concerns in the input field
   - Example: "I have a severe headache and high fever"

2. **Get Analysis**: Click the send button or press Enter
   - The AI extracts specific symptoms using the spaCy NER model
   - Relevant medical information is retrieved using the RAG index

3. **View Results**:
   - **Identified Symptoms**: Shows extracted medical symptoms
   - **Relevant Information**: Displays related medical documents and references

## 🔧 Project Files

- **app.py**: Flask backend server
- **templates/index.html**: Chat interface HTML
- **static/style.css**: Styling for the chat
- **static/script.js**: Frontend JavaScript for chat interactions
- **rag_ner_pipeline.py**: Your existing symptom extraction pipeline
- **model/model-best/**: spaCy NER model
- **rag_index/**: Vector database with medical documents

## 🛠️ Troubleshooting

### Models not loading?
- Ensure you've trained your spaCy model: `python train_model.py`
- Ensure you've built the RAG index: `python build_index.py`

### Port already in use?
Modify `app.py` line with `app.run()`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to another port
```

### CORS errors?
The flask-cors extension is included in requirements.txt - it's already configured.

## 📝 Customization

### Change Colors
Edit `static/style.css`:
- Line 10: Gradient background
- Line 68: Header gradient
- Line 219: Send button gradient

### Adjust Model Parameters
Edit `app.py`:
- Line 10: Change model path if different
- Line 45: Adjust `TOP_K` for number of results

### Modify Welcome Message
Edit `templates/index.html`:
- Lines 30-33: Welcome message content

## 🌐 Deploy to Production

1. Change `debug=False` in app.py
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## 📞 Support

For issues with the medical models and NER pipeline, refer to the original README.md.

Enjoy your MedAI Chat Interface! 🎉
