import axios from 'axios';

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sendMessage = async (
  message: string,
  repoId: string = 'local',
  sessionId?: string
) => {
  const response = await api.post('/chat', {
    message,
    repo_id: repoId,
    session_id: sessionId,
  });

  return response.data;
};

export const askQuestion = async (
  question: string,
  repoId: string = 'local',
  sessionId?: string
) => {
  return sendMessage(question, repoId, sessionId);
};

export const getHealthStatus = async () => {
  const response = await api.get('/health');
  return response.data;
};