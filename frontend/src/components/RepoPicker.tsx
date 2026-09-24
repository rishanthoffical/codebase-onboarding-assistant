import React from 'react';
import type { Repository } from '../services/api';

interface RepoPickerProps {
  repos?: Repository[];
  selectedRepo?: string;
  onRepoChange?: (repo: string) => void;
  loading?: boolean;
}

const RepoPicker: React.FC<RepoPickerProps> = ({
  repos = [],
  selectedRepo = 'local',
  onRepoChange,
  loading = false,
}) => {
  const safeRepos = Array.isArray(repos)
    ? repos
    : [];

  return (
    <div className="repo-picker">
      <label htmlFor="repo-select">
        Repository
      </label>

      <select
        id="repo-select"
        value={selectedRepo}
        disabled={loading}
        onChange={(event) => {
          onRepoChange?.(
            event.target.value
          );
        }}
      >
        {safeRepos.length === 0 ? (
          <option value="local">
            local
          </option>
        ) : (
          safeRepos.map((repo) => (
            <option
              key={repo.repo_id}
              value={repo.repo_id}
            >
              {repo.name}
            </option>
          ))
        )}
      </select>
    </div>
  );
};

export default RepoPicker;
