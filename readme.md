# Code Assistant RAG

## Overview

The Code Assistant RAG (Retrieval-Augmented Generation) is a Streamlit application designed to assist software engineers by providing intelligent code-related answers. It leverages advanced AI models to analyze user queries and retrieve relevant code snippets and documentation from a specified codebase.

## Features

- **Contextual Code Assistance**: The app uses a retrieval-augmented generation approach to provide answers based on the context of the codebase.
- **Multi-Language Support**: Supports various programming languages including Python, Java, JavaScript, C++, and more.
- **User-Friendly Interface**: Built with Streamlit, the app offers an intuitive interface for users to input their coding questions.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables for the AI model and project configuration.

## Usage

1. Run the application:
   ```bash
   streamlit run app/main.py
   ```
2. Open your web browser and navigate to `http://localhost:8501`.
3. Enter your coding question in the text area and click "Get Answer" to receive a response.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
