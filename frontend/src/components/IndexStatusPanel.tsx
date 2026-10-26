import React, { useEffect, useState } from 'react';
import {
  getApiErrorMessage,
  getHealthStatus,
  getRepositoryIndexStatus,
  getReadinessStatus,
} from '../services/api';

interface HealthStatus {
  status?: string;
  [key: string]: unknown;
}

interface IndexStatusPanelProps {
  repoId: string;
  repoName?: string;
}

const IndexStatusPanel: React.FC<IndexStatusPanelProps> = ({
  repoId,
  repoName,
}) => {
  const [healthStatus, setHealthStatus] =
    useState<HealthStatus | null>(null);

  const [healthError, setHealthError] = useState('');
  const [readinessStatus, setReadinessStatus] =
    useState('Checking...');
  const [repositoryStatus, setRepositoryStatus] =
    useState('Checking...');
  const [vectorIndexStatus, setVectorIndexStatus] =
    useState('Checking...');
  const [loading, setLoading] = useState(true);

  const fetchIndexStatus = async () => {
    const [
      healthResult,
      readinessResult,
      indexResult,
    ] = await Promise.allSettled([
      getHealthStatus(),
      getReadinessStatus(),
      getRepositoryIndexStatus(repoId),
    ]);

    if (healthResult.status === 'fulfilled') {
      setHealthStatus(healthResult.value);
      setHealthError('');
    } else {
      setHealthStatus(null);
      setHealthError(
        getApiErrorMessage(
          healthResult.reason,
          'Backend health check failed.'
        )
      );
    }

    if (readinessResult.status === 'fulfilled') {
      setReadinessStatus(
        readinessResult.value.status === 'ready'
          ? 'Available'
          : 'Not ready'
      );
    } else {
      setReadinessStatus(
        `Unavailable: ${getApiErrorMessage(
          readinessResult.reason,
          'Readiness check failed.'
        )}`
      );
    }

    if (indexResult.status === 'fulfilled') {
      setRepositoryStatus(
        indexResult.value.repository_status
      );
      setVectorIndexStatus(
        `${indexResult.value.vector_count} vectors`
      );
    } else {
      const message = getApiErrorMessage(
        indexResult.reason,
        'Repository index status check failed.'
      );
      setRepositoryStatus(`Unavailable: ${message}`);
      setVectorIndexStatus('Unavailable');
    }

    setLoading(false);
  };

  useEffect(() => {
    void fetchIndexStatus();

    const interval = setInterval(() => {
      void fetchIndexStatus();
    }, 5000);

    return () => {
      clearInterval(interval);
    };
  }, [repoId]);

  return (
    <div className="index-status-panel">
      <h2>Index Status</h2>

      {loading && (
        <p>Checking backend status...</p>
      )}

      {!loading && (
        <div>
          <p>
            <strong>Backend:</strong>{' '}
            {healthStatus?.status ||
              (healthError ? `Unavailable (${healthError})` : 'Unavailable')}
          </p>

          <p>
            <strong>Repository:</strong> {repoName || repoId}
          </p>

          <p>
            <strong>Indexing:</strong> {repositoryStatus}
          </p>

          <p>
            <strong>Vector Index:</strong> {vectorIndexStatus}
          </p>

          <p>
            <strong>Dependencies:</strong> {readinessStatus}
          </p>
        </div>
      )}
    </div>
  );
};

export default IndexStatusPanel;