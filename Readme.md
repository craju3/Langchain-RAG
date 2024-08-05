# LangchainRAG

LangchainRAG is a retrieval-augmented generation (RAG) chatbot that uses FAISS for document indexing and embedding retrieval. It employs a SentenceTransformer model to convert documents and queries into embeddings and uses these embeddings to find the most relevant documents for a given query. The relevant documents are then passed to a generative AI model (OpenAI's GPT-3.5) to generate a natural language response.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/LangchainRAG.git
    cd LangchainRAG
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Set your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY='your-openai-api-key'  # On Windows, use `set OPENAI_API_KEY=your-openai-api-key`
    ```

## Usage

1. Run the Flask application:

    ```bash
    python app.py
    ```

2. Make a POST request to the `/ask` endpoint with your query. For example:

    ```bash
    curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d '{"question": "What is the leave policy?"}'
    ```

## Project Structure

<ul>
  <li><strong>LangchainRAG/</strong></li>
    <ul>
      <li><strong>data/</strong></li>
        <ul>
          <li>code_of_conduct.txt</li>
          <li>it_security_policy.txt</li>
          <li>leave_policy.txt</li>
          <li>remote_work_policy.txt</li>
        </ul>
      <li>app.py</li>
      <li>chatbot.py</li>
      <li>requirements.txt</li>
      <li>README.md</li>
    </ul>
</ul>


- `data/`: Contains the text files used for document embedding and retrieval.
- `app.py`: The Flask application that serves the chatbot.
- `chatbot.py`: Contains the logic for embedding documents, performing semantic search using FAISS, and generating responses with GPT-3.5.
- `requirements.txt`: Lists the dependencies required for the project.
- `README.md`: Provides information about the project.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
