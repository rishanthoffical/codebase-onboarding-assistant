# My Codebase Onboarding Assistant

## Overview
The My Codebase Onboarding Assistant is a tool designed to help engineers quickly understand and navigate complex codebases. By leveraging a combination of natural language processing and code analysis, it provides contextual answers to questions about the code, documentation, and git history.

## Features
- **Natural Language Queries**: Ask questions in plain English and receive detailed answers with citations.
- **Real-time Indexing**: The tool automatically updates its index as the codebase changes, ensuring that answers are always based on the most current information.
- **Security Guardrails**: The system scans for sensitive information and insecure coding practices, providing warnings when necessary.
- **Multi-repo Support**: Easily switch between different code repositories within the same interface.

## Tech Stack
- **Backend**: FastAPI (async), Celery (task queue), Redis (broker), Chroma (vector store)
- **Frontend**: React (single-page application)
- **Embedding Model**: Sentence Transformers (local) with the option to use API-based models
- **Containerization**: Docker Compose for easy deployment

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-codebase-onboarding-assistant
   ```

2. Set up the backend:
   - Navigate to the `backend` directory.
   - Create a `.env` file based on the `.env.example` provided.
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```

3. Set up the frontend:
   - Navigate to the `frontend` directory.
   - Install dependencies:
     ```
     npm install
     ```

4. Run the application:
   ```
   docker-compose up
   ```

## Usage
- Access the application via your web browser at `http://localhost:3000`.
- Use **Add a codebase** to upload a `.zip` archive, individual text/code files, or a project folder. The app indexes the selection and automatically generates a beginner-friendly overview with file citations. The defaults allow up to 5,000 files, 500 MB per upload (including extracted ZIP contents), and 10 MB per individual source file; configure these with `MAX_FILES`, `MAX_UPLOAD_SIZE_MB`, and `MAX_SINGLE_FILE_MB` in `backend/.env`. Common runtime, build, cache, and credential files are excluded. For `.rar` or `.7z` archives, extract the folder first or convert the archive to `.zip`.
- Use the chat interface to ask questions about the codebase.
- Monitor indexing status and guardrail warnings in real-time.

## Testing
- Unit tests are located in the `backend/tests/unit` directory.
- Integration tests can be found in the `backend/tests/integration` directory.
- To run tests, use:
  ```
  pytest
  ```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.