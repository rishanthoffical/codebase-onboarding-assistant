import React from 'react';

interface CitationChipProps {
    filePath: string;
    lineStart: number;
    lineEnd: number;
    onClick: () => void;
}

const CitationChip: React.FC<CitationChipProps> = ({ filePath, lineStart, lineEnd, onClick }) => {
    return (
        <div className="citation-chip" onClick={onClick}>
            <span>{`${filePath}:${lineStart}-${lineEnd}`}</span>
        </div>
    );
};

export default CitationChip;