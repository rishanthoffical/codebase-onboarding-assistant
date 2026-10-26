
import React from 'react';

interface ToolCall {
  tool?: string;
  reason?: string;
  output?: string;
}

interface JobTracePanelProps {
  toolCalls?: Array<ToolCall | string>;
}

const JobTracePanel: React.FC<JobTracePanelProps> = ({
  toolCalls = [],
}) => {
  const safeToolCalls = Array.isArray(toolCalls)
    ? toolCalls
    : [];

  return (
    <div className="job-trace-panel">
      <h2>Job Trace</h2>

      {safeToolCalls.length === 0 ? (
        <p>No tool calls yet.</p>
      ) : (
        <ul>
          {safeToolCalls.map((call, index) => {
            const toolName =
              typeof call === 'string'
                ? call
                : call.tool;

            const reason =
              typeof call === 'string'
                ? undefined
                : call.reason;

            const output =
              typeof call === 'string'
                ? undefined
                : call.output;

            return (
              <li key={`${toolName || 'tool'}-${index}`}>
                <div>
                  <strong>Tool:</strong>{' '}
                  {toolName || 'Unknown tool'}
                </div>

                {reason && (
                  <div>
                    <strong>Reason:</strong>{' '}
                    {reason}
                  </div>
                )}

                {output && (
                  <div>
                    <strong>Output:</strong>{' '}
                    {output}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};

export default JobTracePanel;

