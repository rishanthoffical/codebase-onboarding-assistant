import axios from 'axios';
import { zipSync } from 'fflate';

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000';

const API_TOKEN =
  import.meta.env.VITE_API_TOKEN || '';

const getAuthHeaders = (): Record<string, string> => {
  if (!API_TOKEN) {
    return {};
  }

  return {
    Authorization: 'Bearer ' + API_TOKEN,
  };
};

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    ...getAuthHeaders(),
  },
});

export const getApiErrorMessage = (
  error: unknown,
  fallback: string
): string => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;

    if (typeof detail === 'string' && detail.trim()) {
      return detail;
    }

    if (
      error.response?.status === 401 ||
      error.response?.status === 403
    ) {
      return 'Authentication failed. Verify VITE_API_TOKEN matches the backend AUTH_SECRET.';
    }

    if (!error.response) {
      return `Unable to connect to the backend at ${API_BASE_URL}.`;
    }

    return `${fallback} (HTTP ${error.response.status}).`;
  }

  if (
    error instanceof TypeError &&
    error.message.toLowerCase() === 'failed to fetch'
  ) {
    return `Unable to connect to the backend at ${API_BASE_URL}.`;
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return fallback;
};

export interface Repository {
  repo_id: string;
  name: string;
  path: string;
  current_commit?: string | null;
  last_indexed_commit?: string | null;
  status: string;
}

export interface RepositoryIndexStatus {
  repo_id: string;
  vector_count: number;
  index_freshness: 'indexed' | 'empty' | string;
  repository_status: string;
}

export interface ChatResponse {
  answer: string;
  citations?: string[];
  tool_calls_made?: string[];
  confidence?: number | null;
  next_steps_suggestion?: string | null;
}

export interface HealthStatus {
  status: string;
  [key: string]: unknown;
}

export interface ChatHistoryMessage {
  role: 'user' | 'assistant' | string;
  content: string;
  citations?: string[];
  confidence?: number | null;
  next_steps_suggestion?: string | null;
  tool_calls_made?: string[];
}

export interface ChatHistoryResponse {
  session_id: string;
  messages: ChatHistoryMessage[];
}

export const sendMessage = async (
  message: string,
  repoId: string = 'local',
  sessionId?: string
): Promise<ChatResponse> => {
  const response = await api.post(
    '/chat',
    {
      message,
      repo_id: repoId,
      session_id: sessionId,
    },
    {
      headers: {
        ...getAuthHeaders(),
      },
    }
  );

  return response.data;
};

export const getRepositories = async (): Promise<
  Repository[]
> => {
  const response = await api.get(
    '/repos',
    {
      headers: {
        ...getAuthHeaders(),
      },
    }
  );

  const data = response.data;

  if (Array.isArray(data)) {
    return data;
  }

  if (
    data &&
    Array.isArray(data.repositories)
  ) {
    return data.repositories;
  }

  return [];
};

export const getRepositoryIndexStatus = async (
  repoId: string
): Promise<RepositoryIndexStatus> => {
  const response = await api.get(
    `/repos/${encodeURIComponent(repoId)}/index-status`,
    {
      headers: {
        ...getAuthHeaders(),
      },
    }
  );

  return response.data;
};

export const uploadProject = async (
  file: Blob,
  filename: string
): Promise<{
  repository: Repository;
  job_id: string;
  excluded_files?: number;
}> => {
  const response = await api.post(
    '/repos/upload',
    file,
    {
      params: { filename },
      headers: {
        ...getAuthHeaders(),
        'Content-Type': 'application/octet-stream',
      },
    }
  );

  return response.data;
};

export const uploadProjectFiles = async (
  files: File[]
): Promise<{
  repository: Repository;
  job_id: string;
  excluded_files?: number;
}> => {
  const entries: Record<string, Uint8Array> = {};

  for (const [index, file] of files.entries()) {
    let relativePath = file.webkitRelativePath || file.name;

    if (entries[relativePath]) {
      relativePath = `additional-files/${index + 1}/${file.name}`;
    }

    entries[relativePath] = new Uint8Array(
      await file.arrayBuffer()
    );
  }

  const firstRelativePath =
    files[0]?.webkitRelativePath || '';
    const projectName =
    firstRelativePath.split('/')[0] || 'uploaded-project';

  const zipBytes = zipSync(entries, { level: 1 });

  const archive = new Blob(
    [zipBytes],
    { type: 'application/zip' }
  );
  return uploadProject(
    archive,
    `${projectName}.zip`
  );
};

export const getJobStatus = async (
  jobId: string
): Promise<{
  status?: string;
  state?: string;
  meta?: { error?: string };
  error?: string;
}> => {
  const response = await api.get(
    `/status/${encodeURIComponent(jobId)}`,
    {
      headers: {
        ...getAuthHeaders(),
      },
    }
  );

  return response.data;
};

export const getHealthStatus = async (): Promise<
  HealthStatus
> => {
  const response = await api.get(
    '/health',
    {
      headers: {
        ...getAuthHeaders(),
      },
    }
  );

  return response.data;
};

export const getReadinessStatus = async (): Promise<
  HealthStatus
> => {
  const response = await api.get(
    '/readiness',
    {
      headers: {
        ...getAuthHeaders(),
      },
    }
  );

  return response.data;
};

export const getChatHistory = async (
  sessionId: string,
  repoId: string = 'local'
): Promise<ChatHistoryResponse> => {
  const response = await api.get(
    `/chat/history/${encodeURIComponent(sessionId)}`,
    {
      params: { repo_id: repoId },
      headers: {
        ...getAuthHeaders(),
      },
    }
  );

  return response.data;
};

export interface StreamEvent {
  type:
    | 'start'
    | 'tool'
    | 'content'
    | 'warning'
    | 'citation'
    | 'completion'
    | 'error'
    | string;

  repo_id?: string;

  // Tool event fields
  name?: string;
  status?: string;
  tool?: string;

  // Content/warning fields
  content?: string;
  warning?: string;

  // Citation fields
  citation?: string;

  // Completion fields
  answer?: string;
  citations?: string[];
  tool_calls_made?: string[];
  confidence?: number | null;
  next_steps_suggestion?: string | null;

  // Error fields
  message?: string;
  error?: string;
}

export interface StreamMessageCallbacks {
  onStart?: (
    event: StreamEvent
  ) => void;

  onTool?: (
    event: StreamEvent
  ) => void;

  onContent?: (
    content: string,
    event: StreamEvent
  ) => void;

  onWarning?: (
    warning: string,
    event: StreamEvent
  ) => void;

  onCitation?: (
    citation: string,
    event: StreamEvent
  ) => void;

  onCompletion?: (
    event: StreamEvent
  ) => void;

  onError?: (
    error: Error,
    event?: StreamEvent
  ) => void;
}

/**
 * Stream assistant responses from the FastAPI
 * SSE endpoint.
 */
export const streamMessage = async (
  message: string,
  repoId: string = 'local',
  sessionId?: string,
  callbacks: StreamMessageCallbacks = {}
): Promise<void> => {
  const response = await fetch(
    `${API_BASE_URL}/chat/stream`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        ...getAuthHeaders(),
      },
      body: JSON.stringify({
        message,
        repo_id: repoId,
        session_id: sessionId,
      }),
    }
    
  );

  if (!response.ok) {
    let detail =
      `Streaming request failed with status ${response.status}.`;

    try {
      const errorBody =
        await response.json();

      if (
        errorBody &&
        typeof errorBody.detail ===
          'string'
      ) {
        detail = errorBody.detail;
      }
    } catch {
      // Keep default error message.
    }

    const error = new Error(detail);

    callbacks.onError?.(error);

    throw error;
  }

  if (!response.body) {
    const error = new Error(
      'Streaming response body is unavailable.'
    );

    callbacks.onError?.(error);

    throw error;
  }

  const reader =
    response.body.getReader();

  const decoder =
    new TextDecoder('utf-8');

  let buffer = '';

  const processEvent = (
    rawEvent: string
  ) => {
    const lines =
      rawEvent.split(/\r?\n/);

    const eventLine =
      lines.find((line) =>
        line.startsWith('event:')
      );

    const eventType =
      eventLine
        ?.slice('event:'.length)
        .trim() || '';

    const dataLines =
      lines
        .filter((line) =>
          line.startsWith('data:')
        )
        .map((line) =>
          line
            .slice('data:'.length)
            .trim()
        );

    if (
      dataLines.length === 0
    ) {
      return;
    }

    const dataText =
      dataLines.join('\n');

    if (!dataText) {
      return;
    }

    let data: Record<
      string,
      unknown
    >;

    try {
      data = JSON.parse(
        dataText
      );
    } catch (error) {
      console.warn(
        'Unable to parse SSE response data:',
        error
      );

      return;
    }

    const event: StreamEvent = {
      ...(data as Partial<StreamEvent>),
      type: eventType,
    };

    switch (event.type) {
      case 'start':
        callbacks.onStart?.(
          event
        );
        break;

      case 'tool':
        callbacks.onTool?.(
          event
        );
        break;

      case 'content':
      case 'message':
        if (
          typeof event.content ===
          'string'
        ) {
          callbacks.onContent?.(
            event.content,
            event
          );
        }
        break;

      case 'warning':
        if (
          typeof event.warning ===
          'string'
        ) {
          callbacks.onWarning?.(
            event.warning,
            event
          );
        }
        break;

      case 'citation':
        if (
          typeof event.citation ===
          'string'
        ) {
          callbacks.onCitation?.(
            event.citation,
            event
          );
        }
        break;

      case 'citations':
        if (Array.isArray(event.citations)) {
          for (const citation of event.citations) {
            if (typeof citation === 'string') {
              callbacks.onCitation?.(citation, event);
            }
          }
        }
        break;
      case 'completion':
      case 'complete':
        callbacks.onCompletion?.(
          event
        );
        break;

      case 'error': {
        const error = new Error(
          typeof event.message ===
          'string'
            ? event.message
            : typeof event.error === 'string'
              ? event.error
              : 'The streaming request failed.'
        );

        callbacks.onError?.(
          error,
          event
        );

        throw error;
      }

      default:
        console.warn(
          'Unknown SSE event type:',
          event.type
        );
    }
  };

  try {
    while (true) {
      const {
        value,
        done,
      } = await reader.read();

      if (done) {
        break;
      }

      buffer +=
        decoder.decode(
          value,
          {
            stream: true,
          }
        );

      const events =
        buffer.split(
          /\r?\n\r?\n/
        );

      buffer =
        events.pop() || '';

      for (
        const rawEvent of events
      ) {
        if (
          rawEvent.trim()
        ) {
          processEvent(
            rawEvent
          );
        }
      }
    }

    buffer +=
      decoder.decode();

    const remaining =
      buffer.trim();

    if (remaining) {
      processEvent(
        remaining
      );
    }
  } catch (error) {
    const streamError =
      error instanceof Error
        ? error
        : new Error(
            'The streaming request failed.'
          );

    callbacks.onError?.(
      streamError
    );

    throw streamError;
  }
};
