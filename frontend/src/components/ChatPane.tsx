import React, { useState } from 'react';
import { sendMessage } from '../services/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatPaneProps {
  repoId?: string;
  sessionId?: string;
  onToolCalls?: (calls: any[]) => void;
}

const ChatPane: React.FC<ChatPaneProps> = ({
  repoId = 'local',
  sessionId,
  onToolCalls,
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const [currentSessionId] = useState<string>(
    () => sessionId || `session-${Date.now()}`
  );

  const handleSend = async () => {
    const question = input.trim();

    if (!question || loading) {
      return;
    }

    // Add user's message to chat
    const userMessage: Message = {
      role: 'user',
      content: question,
    };

    setMessages((previous) => [
      ...previous,
      userMessage,
    ]);

    setInput('');
    setLoading(true);

    try {
      // Send question to FastAPI backend
      const response = await sendMessage(
        question,
        repoId,
        currentSessionId
      );

      // Send tool calls to Job Trace
      if (onToolCalls) {
        onToolCalls(Array.isArray(response?.tool_calls_made) ? response.tool_calls_made : []);
      }

      // Add backend response
      const assistantMessage: Message = {
        role: 'assistant',
        content:
          response?.answer ||
          'No answer was returned by the backend.',
      };

      setMessages((previous) => [
        ...previous,
        assistantMessage,
      ]);
    } catch (error) {
      console.error(
        'Error sending message to backend:',
        error
      );

      const errorMessage: Message = {
        role: 'assistant',
        content:
          'Unable to connect to the backend. Please make sure FastAPI is running on http://127.0.0.1:8001.',
      };

      setMessages((previous) => [
        ...previous,
        errorMessage,
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement>
  ) => {
    if (event.key === 'Enter') {
      handleSend();
    }
  };

  return (
    <div className="chat-pane">

      {/* Chat Messages */}
      <div className="chat-messages">

        {messages.length === 0 ? (
          <div className="empty-chat">
            Ask a question about the codebase.
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.role}`}
            >
              <strong>
                {message.role === 'user'
                  ? 'You'
                  : 'Assistant'}
              </strong>

              <p>{message.content}</p>
            </div>
          ))
        )}

        {/* Loading message */}
        {loading && (
          <div className="message assistant">
            <strong>Assistant</strong>
            <p>Thinking...</p>
          </div>
        )}

      </div>

      {/* Chat Input */}
      <div className="chat-input-area">

        <input
          type="text"
          value={input}
          placeholder="Ask about the codebase..."
          onChange={(event) =>
            setInput(event.target.value)
          }
          onKeyDown={handleKeyDown}
          disabled={loading}
        />

        <button
          type="button"
          onClick={handleSend}
          disabled={loading || !input.trim()}
        >
          {loading ? 'Sending...' : 'Send'}
        </button>

      </div>

    </div>
  );
};

export default ChatPane;
