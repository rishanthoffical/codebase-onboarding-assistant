# Backend Codebase Onboarding Assistant

This project is designed to assist engineers in navigating and understanding a codebase through a robust onboarding assistant. It leverages a combination of FastAPI, Celery, and a vector store to provide real-time insights and answers to queries about the codebase.

## Project Structure

The backend is organized into several modules, each serving a specific purpose:

- **app**: Contains the core application logic, including the agent, API, configuration, guardrails, indexer, models, services, tasks, and tools.
- **tests**: Includes unit and integration tests to ensure the functionality and reliability of the application.
- **.env.example**: Provides an example environment configuration for setting up the application.
- **Dockerfile**: Contains instructions for building the Docker image for the backend.
- **requirements.txt**: Lists the Python dependencies required for the backend.
- **pytest.ini**: Configuration settings for running tests with pytest.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-codebase-onboarding-assistant/backend
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Copy `.env.example` to `.env` and fill in the necessary values.

## Running the Application

To run the backend application, use the following command:
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Running Tests

To run the tests, execute:
```
pytest
```

## Docker

To build and run the application using Docker, use:
```
docker-compose up --build
```

## Usage

Once the application is running, you can interact with the API at `http://localhost:8000`. The API provides endpoints for querying the codebase, managing repositories, and retrieving chat history.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.