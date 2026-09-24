import React, { useState } from 'react';
import ChatPane from './components/ChatPane';
import IndexStatusPanel from './components/IndexStatusPanel';
import RepoPicker from './components/RepoPicker';
import JobTracePanel from './components/JobTracePanel';
import GuardrailBanner from './components/GuardrailBanner';
import './styles.css';

interface ToolCall {
    tool?: string;
    reason?: string;
    output?: string;
}

const App: React.FC = () => {
    const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);

    return (
        <div className="app">
            <GuardrailBanner />
            <RepoPicker />
            <IndexStatusPanel />
            <ChatPane onToolCalls={(calls) => setToolCalls(calls)} />
            <JobTracePanel toolCalls={toolCalls} />
        </div>
    );
};

export default App;
