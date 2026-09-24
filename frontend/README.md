# Frontend Codebase Onboarding Assistant

This project is a frontend application designed to assist engineers in navigating and understanding a codebase. It provides a user-friendly interface for querying code-related questions and retrieving relevant information from the backend.

## Getting Started

To get started with the frontend application, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd my-codebase-onboarding-assistant/frontend
   ```

2. **Install Dependencies**
   Ensure you have Node.js installed, then run:
   ```bash
   npm install
   ```

3. **Run the Development Server**
   Start the development server with:
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:3000`.

## Project Structure

- **public/**: Contains public assets such as images and icons.
- **src/**: Contains the source code for the application.
  - **components/**: React components used throughout the application.
  - **services/**: API service functions for interacting with the backend.
  - **App.tsx**: Main application component.
  - **main.tsx**: Entry point for the React application.
  - **styles.css**: Styles for the frontend application.

## Features

- **Chat Interface**: Users can ask questions about the codebase and receive answers with citations.
- **Guardrail Indicators**: Displays warnings for insecure patterns or redacted information.
- **Index Status Panel**: Shows the current indexing status of the codebase.
- **Repo Picker**: Allows users to switch between different repositories.
- **Job Trace Viewer**: Displays the sequence of tool calls made during the query process.

## Environment Variables

Make sure to set the following environment variables in your `.env` file:

- `REACT_APP_API_URL`: The URL of the backend API.

## Testing

To run tests, use the following command:
```bash
npm test
```

## Deployment

For production deployment, build the application using:
```bash
npm run build
```
The production-ready files will be available in the `dist` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.