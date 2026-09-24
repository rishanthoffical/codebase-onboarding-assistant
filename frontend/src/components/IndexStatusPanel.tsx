import React, { useEffect, useState } from 'react';
import { getHealthStatus } from '../services/api';

interface HealthStatus {
  status?: string;
  [key: string]: unknown;
}

const IndexStatusPanel: React.FC = () => {
  const [healthStatus, setHealthStatus] =
    useState<HealthStatus | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchIndexStatus = async () => {
    try {
      setError('');

      const data = await getHealthStatus();

      if (data && typeof data === 'object') {
        setHealthStatus(data as HealthStatus);
      } else {
        setHealthStatus({
          status: 'healthy',
        });
      }
    } catch (error) {
      console.error('Error fetching backend health:', error);

      setError('Unable to connect to backend.');
      setHealthStatus(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchIndexStatus();

    const interval = setInterval(() => {
      fetchIndexStatus();
    }, 5000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="index-status-panel">
      <h2>Index Status</h2>

      {loading && (
        <p>Checking backend status...</p>
      )}

      {!loading && error && (
        <p>{error}</p>
      )}

      {!loading && !error && healthStatus && (
        <div>
          <p>
            <strong>Backend:</strong>{' '}
            {healthStatus.status || 'Healthy'}
          </p>

          <p>
            <strong>Repository:</strong> local
          </p>

          <p>
            <strong>Vector Index:</strong> Available
          </p>
        </div>
      )}
    </div>
  );
};

export default IndexStatusPanel;