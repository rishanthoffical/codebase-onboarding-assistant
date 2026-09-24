import React from 'react';

interface JobTracePanelProps {
  toolCalls?: string[];
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
          {safeToolCalls.map((call, index) => (
            <li key={index}>
              <strong>Tool:</strong> {call}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default JobTracePanel;
