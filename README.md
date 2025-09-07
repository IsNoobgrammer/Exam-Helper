# 🌟 Bhala Manus: No BackLog Abhiyan 🌟

<p align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT81LIjm080VFPYpMizkeSMnjfNENB0poYT8Q&s" alt="Project Logo">
</p>

> Padh le yaar... (Study hard, buddy!)

### [**Hosted App**](https://good-boy.streamlit.app/)

**Bhala Manus** is your AI-powered study companion, meticulously crafted to help you conquer your computer science courses and banish the fear of backlogs. This powerful tool integrates cutting-edge language models, a robust vector database, and real-time web search to deliver comprehensive, digestible explanations that make learning a joyful experience.

## ✨ Key Features

### 🧠 Multi-Source Contextual Understanding
Bhala Manus intelligently gathers information from various sources to provide the most relevant and accurate answers:
- **Web Data**: Real-time information retrieval from the internet
- **Documents Data**: Extracts key information from your chosen document sets
- **Chat History**: Maintains context by remembering previous interactions (optional)
- **LLM Data**: Harnesses the power of large language models for insightful responses

### 📚 Document Versatility
Choose between different vector stores (documents) for focused learning on specific topics:
- **cc-docs**: Computer Networks and Cryptography
- **ann-docs**: Artificial Neural Networks
- **dbms-docs**: Database Management Systems

### 🤖 Advanced LLM Selection
Choose between various versions of the Llama 3 Models, each optimized for different needs:
- `llama-3.3-70b-versatile`
- `llama-3.1-70b-versatile`
- `llama-3.1-8b-instant`
- `llama-3.2-90b-vision-preview`

### 🎛️ Enhanced Customization
- **Internet Access**: Decide whether to allow web search during the session
- **Chat History**: Choose to enable or disable the use of previous chat history
- **Document Selection**: Focus on specific subject areas

### 🖼️ Multimedia Query Support
- **Image-Based Question Answering**: Upload images of questions, diagrams, or complex tables and get analyzed, detailed responses
- **YouTube Video Summarization**: Input a YouTube link and get comprehensive summaries tailored to your queries with transcriptions
- **Diagram Generation**: Automatic diagram suggestions for visual learning

### 🎨 Streamlined User Experience
- Intuitive and visually appealing Streamlit-based user interface
- Engaging animated gradient background to make learning fun
- Responsive design with modern UI components

## 🏗️ Project Structure

```
Exam-Helper/
├── main.py                 # Main application entry point
├── utils.py               # Core utility functions
├── ui_components.py       # UI components and styling
├── config_loader.py       # Configuration management
├── config.yaml           # Main configuration file
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation
└── LICENSE            # License file
```

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/IsNoobgrammer/Exam-Helper.git
   cd Exam-Helper
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

5. **Configure API Keys:**
   - **Groq API Key**: Sign up for a free account on [Groq](https://console.groq.com/keys)
   - **Google API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **Pinecone API Key**: Sign up at [Pinecone](https://www.pinecone.io/)
   - **Mistral API Key**: Get from [Mistral AI](https://console.mistral.ai/)

## 🚀 Usage

### Running the Application

1. **Launch the Streamlit app:**
   ```bash
   streamlit run main.py
   ```

2. **Configure Your Environment:**
   - Enter your **Groq API key** in the sidebar
   - Select your preferred **Llama 3 model** for LLM inference
   - Choose your specific **document set** from the dropdown menu
   - Toggle **Internet access** and **chat history** according to your preference

3. **Start Learning!**
   - Type in your questions
   - Upload images of problems or diagrams
   - Input YouTube links for video summaries
   - Experience the power of AI-driven learning

### Configuration Options

The application uses a YAML-based configuration system:

- **API Keys**: Managed through `config.yaml` and environment variables
- **Model Settings**: Default models, temperature settings, and fallback options
- **UI Configuration**: Colors, animations, and styling options
- **Feature Toggles**: Enable/disable specific features

## 🛠️ Dependencies

### Core Libraries
- **LangChain**: Framework for building LLM-powered applications
- **Streamlit**: Web application framework
- **Pinecone**: Vector database for document storage
- **Google Generative AI**: Embedding and vision services

### AI/ML Libraries
- **Groq**: Access to Llama 3 models
- **Mistral AI**: Fallback language model
- **DuckDuckGo Search**: Real-time web search

### Utility Libraries
- **PyYAML**: Configuration management
- **Pydantic**: Data validation
- **Pillow**: Image processing
- **Requests**: HTTP client

## 📁 Configuration Management

### config.yaml Structure
```yaml
api_keys:
  pinecone: "your_key_here"
  google: "your_key_here"
  # ... other keys

app_config:
  page_title: "Bhala Manus"
  default_model: "llama-3.3-70b-versatile"
  # ... other settings

ui_config:
  primary_color: "#47fffc"
  # ... styling options
```

### Environment Variables
Use `.env` file for sensitive configuration:
```env
PINECONE_API_KEY=your_actual_key
GOOGLE_API_KEY=your_actual_key
# ... other keys
```

## 🔧 Development

### Code Organization
- **main.py**: Application entry point and main logic
- **utils.py**: Core functionality (LLM operations, data processing)
- **ui_components.py**: UI components and styling
- **config_loader.py**: Configuration management utilities

### Adding New Features
1. Update configuration in `config.yaml`
2. Add utility functions in `utils.py`
3. Create UI components in `ui_components.py`
4. Integrate in `main.py`

### Testing
```bash
# Run the application in development mode
streamlit run main.py --server.runOnSave true
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update configuration files as needed
- Test your changes thoroughly

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **LangChain Team**: For the amazing framework
- **Streamlit Team**: For the excellent web app framework
- **Groq**: For providing access to Llama models
- **Google AI**: For generative AI services
- **Pinecone**: For vector database services
- **All Contributors**: Thank you for making this project better!

### Current Contributors

<a href="https://github.com/IsNoobgrammer/Exam-Helper/contributors">
  <img src="https://contrib.rocks/image?repo=IsNoobgrammer/Exam-Helper" />
</a>

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/IsNoobgrammer/Exam-Helper/issues)
- **Discussions**: [GitHub Discussions](https://github.com/IsNoobgrammer/Exam-Helper/discussions)
- **Documentation**: This README and inline code documentation

## 🎯 Roadmap

- [ ] Add support for more document formats
- [ ] Implement user authentication
- [ ] Add export functionality for conversations
- [ ] Mobile app development
- [ ] Integration with more LLM providers
- [ ] Advanced analytics and learning insights

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/IsNoobgrammer">IsNoobGrammer</a> and Contributors
</p>

<p align="center">
  <strong>Padh le yaar, backlog se bachne ke liye! 📚✨</strong>
</p>