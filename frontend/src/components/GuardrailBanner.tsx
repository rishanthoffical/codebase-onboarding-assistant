import React from 'react';

interface GuardrailBannerProps {
  message?: string;
  type?: 'warning' | 'error' | 'info';
}

const GuardrailBanner: React.FC<GuardrailBannerProps> = ({
  message,
  type = 'info',
}) => {
  // Do not render an empty banner when there is no guardrail message.
  if (!message) {
    return null;
  }

  const bannerClass = `guardrail-banner ${type}`;

  return (
    <div className={bannerClass}>
      {message}
    </div>
  );
};

export default GuardrailBanner;