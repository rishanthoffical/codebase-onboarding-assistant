import React, {
  useEffect,
  useRef,
  useState,
} from 'react';

import {
  getChatHistory,
  getApiErrorMessage,
  streamMessage,
} from '../services/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: string[];
  confidence?: number | null;
  nextStepsSuggestion?: string | null;
}

interface ChatPaneProps {
  repoId?: string;
  sessionId?: string;
  initialQuestion?: string;
  onToolCalls?: (calls: any[]) => void;
  onResponseComplete?: () => void;
  onResponseError?: () => void;
}

const ChatPane: React.FC<ChatPaneProps> = ({
  repoId = 'local',
  sessionId,
  initialQuestion,
  onToolCalls,
  onResponseComplete,
  onResponseError,
}) => {
   console.log('CHATPANE REPO ID:', repoId);
  const [messages, setMessages] = useState<Message[]>(
    []
  );

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] =
    useState(true);
  const [loadedHistoryKey, setLoadedHistoryKey] =
    useState('');

  const [currentSessionId, setCurrentSessionId] =
    useState<string>(() => {
    if (sessionId) {
      return sessionId;
    }

    const storageKey = 'chat-session-' + repoId;
    const existingSession =
      window.localStorage.getItem(storageKey);

    if (existingSession) {
      return existingSession;
    }

    const newSession = `session-${Date.now()}`;

    window.localStorage.setItem(
      storageKey,
      newSession
    );

    return newSession;
  });

  const [sessionRepoId, setSessionRepoId] =
    useState(repoId);

  const initialPromptSent = useRef('');

  useEffect(() => {
    if (sessionRepoId === repoId) {
      return;
    }

    const storageKey = 'chat-session-' + repoId;
    const existingSession = sessionId
      ? null
      : window.localStorage.getItem(storageKey);
    const nextSession =
      sessionId || existingSession || `session-${Date.now()}`;

    if (!existingSession && !sessionId) {
      window.localStorage.setItem(
        storageKey,
        nextSession
      );
    }

    setCurrentSessionId(nextSession);
    setSessionRepoId(repoId);
  }, [repoId, sessionId, sessionRepoId]);

  useEffect(() => {
    if (sessionRepoId !== repoId) {
      return;
    }

    let cancelled = false;

    const loadHistory = async () => {
      setMessages([]);
      setHistoryLoading(true);

      try {
        const response = await getChatHistory(
          currentSessionId,
          repoId
        );

        if (
          cancelled ||
          !response ||
          !Array.isArray(response.messages)
        ) {
          return;
        }

        const restoredMessages: Message[] =
          response.messages
            .filter(
              (
                message
              ): message is typeof message & {
                role: 'user' | 'assistant';
              } =>
                message.role === 'user' ||
                message.role === 'assistant'
            )
            .map((message) => ({
              role: message.role,
              content: message.content,
              citations:
                message.citations || [],
              confidence:
                message.confidence ?? null,
              nextStepsSuggestion:
                message.next_steps_suggestion ??
                null,
            }));

        setMessages(restoredMessages);
      } catch (error) {
        console.warn(
          'Unable to restore chat history:',
          getApiErrorMessage(
            error,
            'Unable to restore chat history.'
          )
        );
      } finally {
        if (!cancelled) {
          setHistoryLoading(false);
          setLoadedHistoryKey(
            `${repoId}:${currentSessionId}`
          );
        }
      }
    };

    loadHistory();

    return () => {
      cancelled = true;
    };
  }, [currentSessionId, repoId, sessionRepoId]);

  const handleSend = async (
    questionOverride?: string
  ) => {
    const question = (questionOverride ?? input).trim();

    if (
      !question ||
      loading ||
      historyLoading
    ) {
      return;
    }

    const userMessage: Message = {
      role: 'user',
      content: question,
    };

    setMessages((previous) => [
      ...previous,
      userMessage,
      {
        role: 'assistant',
        content: '',
        citations: [],
        confidence: null,
        nextStepsSuggestion: null,
      },
    ]);

    setInput('');
    setLoading(true);

    let assistantContent = '';
    let toolCalls: string[] = [];
    let citations: string[] = [];

   try {
      console.log('BEFORE STREAM MESSAGE:', {
    question,
    repoId,
    currentSessionId,
  });
   console.log('STREAM VALUES:', {
  question: JSON.stringify(question),
  repoId: JSON.stringify(repoId),
  currentSessionId: JSON.stringify(currentSessionId),
});
  await streamMessage(
    question,
    repoId,
    currentSessionId,
        {
          onStart: () => {
            // Stream started.
          },

          onTool: (event) => {
            const toolName =
              event.name || event.tool;

            if (toolName) {
              toolCalls = [
                ...toolCalls,
                toolName,
              ];

              if (onToolCalls) {
                onToolCalls(toolCalls);
              }
            }
          },

          onContent: (content) => {
            assistantContent += content;

            setMessages((previous) => {
              const updated = [...previous];

              const assistantIndex =
                updated.length - 1;

              if (
                updated[assistantIndex] &&
                updated[assistantIndex].role ===
                  'assistant'
              ) {
                updated[assistantIndex] = {
                  ...updated[assistantIndex],
                  content:
                    assistantContent,
                };
              }

              return updated;
            });
          },

          onWarning: (warning) => {
            console.warn(
              'Backend warning:',
              warning
            );
          },

          onCitation: (citation) => {
            if (
              !citations.includes(citation)
            ) {
              citations = [
                ...citations,
                citation,
              ];

              setMessages((previous) => {
                const updated = [...previous];

                const assistantIndex =
                  updated.length - 1;

                if (
                  updated[assistantIndex] &&
                  updated[assistantIndex]
                    .role === 'assistant'
                ) {
                  updated[assistantIndex] = {
                    ...updated[assistantIndex],
                    citations,
                  };
                }

                return updated;
              });
            }
          },

          onCompletion: (event) => {
            const finalAnswer =
              event.answer ||
              assistantContent ||
              'No answer was returned by the backend.';

            const finalCitations =
              Array.isArray(
                event.citations
              )
                ? event.citations
                : citations;

            setMessages((previous) => {
              const updated = [...previous];

              const assistantIndex =
                updated.length - 1;

              if (
                updated[assistantIndex] &&
                updated[assistantIndex].role ===
                  'assistant'
              ) {
                updated[assistantIndex] = {
                  ...updated[assistantIndex],
                  content: finalAnswer,
                  citations:
                    finalCitations,
                  confidence:
                    typeof event.confidence ===
                    'number'
                      ? event.confidence
                      : null,
                  nextStepsSuggestion:
                    event.next_steps_suggestion ||
                    null,
                };
              }

              return updated;
            });

            if (onToolCalls) {
              onToolCalls(
                Array.isArray(
                  event.tool_calls_made
                )
                  ? event.tool_calls_made
                  : toolCalls
              );
            }

            onResponseComplete?.();
          },

          onError: (error) => {
            const message = getApiErrorMessage(
              error,
              'Unable to send the message.'
            );
            console.error('Streaming error:', message);

            setMessages((previous) => {
              const updated = [...previous];

              const assistantIndex =
                updated.length - 1;

              if (
                updated[assistantIndex] &&
                updated[assistantIndex].role ===
                  'assistant'
              ) {
                updated[assistantIndex] = {
                  ...updated[assistantIndex],
                  content: message,
                };
              }

              return updated;
            });
          },
        }
      );
    } catch (error) {
      const message = getApiErrorMessage(
        error,
        'Unable to send the message.'
      );
      console.error('Error streaming message:', message);
      onResponseError?.();

      setMessages((previous) => {
        const updated = [...previous];

        const assistantIndex =
          updated.length - 1;

        if (
          updated[assistantIndex] &&
          updated[assistantIndex].role ===
            'assistant'
        ) {
          updated[assistantIndex] = {
            ...updated[assistantIndex],
            content: message,
          };
        }

        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (
      !initialQuestion ||
      historyLoading ||
      sessionRepoId !== repoId ||
      loadedHistoryKey !== `${repoId}:${currentSessionId}`
    ) {
      return;
    }

    const promptKey = `${repoId}:${currentSessionId}:${initialQuestion}`;

    if (initialPromptSent.current === promptKey) {
      return;
    }

    initialPromptSent.current = promptKey;
    void handleSend(initialQuestion);
  }, [
    currentSessionId,
    handleSend,
    historyLoading,
    initialQuestion,
    loadedHistoryKey,
    repoId,
    sessionRepoId,
  ]);

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement>
  ) => {
    if (event.key === 'Enter') {
      handleSend();
    }
  };

  const formatCitation = (
    citation: string
  ) => {
    return citation.replace(
      /^(.+):(\d+)-(\d+)$/,
      '$1:$2-$3'
    );
  };

  return (
    <div className="chat-pane">
      <div className="chat-messages">
        {historyLoading ? (
          <div className="empty-chat">
            Loading conversation history...
          </div>
        ) : messages.length === 0 ? (
          <div className="empty-chat">
            Ask a question about the codebase.
          </div>
        ) : (
          messages.map(
            (message, index) => (
              <div
                key={`${currentSessionId}-${index}`}
                className={`message ${message.role}`}
              >
                <strong>
                  {message.role ===
                  'user'
                    ? 'You'
                    : 'Assistant'}
                </strong>

                <p style={{ whiteSpace: 'pre-wrap' }}>
                  {message.content ||
                    (message.role ===
                    'assistant'
                      ? 'Thinking...'
                      : '')}
                </p>

                {message.role ===
                  'assistant' &&
                  message.citations &&
                  message.citations
                    .length > 0 && (
                    <div className="citations">
                      <strong>
                        Sources
                      </strong>

                      {message.citations.map(
                        (
                          citation,
                          citationIndex
                        ) => (
                          <div
                            key={`${citation}-${citationIndex}`}
                            className="citation"
                          >
                            📄{' '}
                            {formatCitation(
                              citation
                            )}
                          </div>
                        )
                      )}
                    </div>
                  )}

                {message.role ===
                  'assistant' &&
                  message.confidence !==
                    null &&
                  message.confidence !==
                    undefined && (
                    <div className="confidence">
                      Confidence:{' '}
                      {Math.round(
                        message.confidence *
                          100
                      )}
                      %
                    </div>
                  )}

                {message.role ===
                  'assistant' &&
                  message.nextStepsSuggestion && (
                    <div className="next-step">
                      <strong>
                        Next step:
                      </strong>{' '}
                      {
                        message.nextStepsSuggestion
                      }
                    </div>
                  )}
              </div>
            )
          )
        )}
      </div>

      <div className="chat-input-area">
        <input
          type="text"
          value={input}
          placeholder="Ask about the codebase..."
          onChange={(event) =>
            setInput(event.target.value)
          }
          onKeyDown={handleKeyDown}
          disabled={
            loading ||
            historyLoading
          }
        />

        <button
          type="button"
          onClick={() => handleSend()}
          disabled={
            loading ||
            historyLoading ||
            !input.trim()
          }
        >
          {loading
            ? 'Streaming...'
            : historyLoading
            ? 'Loading...'
            : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default ChatPane;
