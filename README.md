# 🤖 QueryBot: AI Information Extraction Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An intelligent agent that automates web searches and information extraction from datasets using LLMs and modern AI tools. Perfect for researchers, analysts, and data enthusiasts who need to enrich their datasets with web-sourced information.

## 🌟 Features

- 📊 **Smart Data Upload**: Support for CSV files with intuitive column selection
- 🔍 **Dynamic Search Queries**: Customizable prompt templates for flexible information extraction
- 🧠 **AI-Powered Extraction**: Leverages Google's Gemini Pro for intelligent data parsing
- 📈 **Real-time Progress Tracking**: Visual feedback on extraction progress
- 📥 **Multiple Export Options**: Download results in CSV or Excel format
- 🎨 **Modern Dark Theme UI**: Clean, responsive interface built with Streamlit
- 🛡️ **Robust Error Handling**: Graceful handling of API failures and rate limits

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.8+ installed on your system. All dependencies are listed in `requirements.txt`.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Query-bot.git
cd querybot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
# Create a .env file and add your API keys in the 30th and 31st line of the app.py file
SERPAPI_KEY=your_serp_api_key
GEMINI_KEY=your_gemini_api_key
```

4. Run the application:
```bash
streamlit run app.py
```

## 💡 Usage Guide

1. **Upload Your Data**
   - Click the "Choose a CSV file" button to upload your dataset
   - Preview your data in the expandable section

2. **Configure Your Search**
   - Select the column containing your target entities
   - Enter a search prompt using `{entity}` as a placeholder
   - Example: "Get the email and headquarters location for {entity}"

3. **Process & Extract**
   - Click "Process Data" to start the extraction
   - Monitor progress in real-time
   - View success rates and failed extractions

4. **Export Results**
   - Download processed data in CSV or Excel format
   - All results include timestamps and status indicators

## ⚙️ Configuration Options

Access additional settings in the sidebar:

- **Batch Size**: Control how many entities are processed simultaneously (1-10)
- **Max Retries**: Set maximum retry attempts for failed extractions (1-5)
- **Temperature**: Adjust LLM creativity level (0.0-1.0)
- **Confidence Threshold**: Set minimum confidence score for extractions (0.0-1.0)

## 🛠️ Technical Implementation

- **Frontend**: Streamlit with custom CSS for dark theme
- **Search**: SerpAPI integration for web searching
- **LLM**: Google's Gemini Pro via LangChain
- **Data Processing**: Pandas for efficient data handling
- **Concurrency**: ThreadPoolExecutor for parallel processing
- **Error Handling**: Comprehensive retry mechanism with logging

## ✨ Advanced Features

### Enhanced Processing Capabilities
- **Batch Processing**: Configurable batch sizes (1-10) for optimal performance
- **Retry Mechanism**: Adjustable retry attempts (1-5) for failed extractions
- **Rate Limiting**: Smart API call management to prevent rate limiting issues
- **Concurrent Processing**: Parallel execution using ThreadPoolExecutor
- **Text Splitting**: Advanced text handling for large content processing

### Advanced Results Analysis
- **Success Metrics**: Real-time tracking of successful extractions
- **Failure Analysis**: Detailed tracking of failed entries
- **Processing History**: Session-based history tracking
- **Performance Metrics**: Success rates and processing speed analysis
- **Data Preview**: Quick view of dataset statistics and unique entries

### Enhanced Export System
- **Multiple Formats**: Export to both CSV and Excel formats
- **Smart Naming**: Timestamp-based file naming for better organization
- **Optimized Structure**: Intelligent column ordering in exports
- **Excel Enhancement**: Professional Excel exports using xlsxwriter

### UI/UX Improvements
- **Dark Theme**: Professional dark mode interface
- **Progress Tracking**: Visual progress bars and status updates
- **Expandable Sections**: Organized information display
- **Dual-Column Layout**: Efficient space utilization
- **Interactive Metrics**: Real-time performance indicators

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google's Gemini Pro](https://deepmind.google/technologies/gemini/)
- Search capability provided by [SerpAPI](https://serpapi.com/)
- LLM integration via [LangChain](https://python.langchain.com/)

## 📬 Contact

For questions and feedback:
- 📧 Email: nitinsagar2004@gmail.com
- 🌐 LinkedIn: [Add me to your network via this link!](https://linkedin.com/in/nitin-sagar-boyeena)

---

Made with ❤️ by [Nitin] [Sparkience-AI]
