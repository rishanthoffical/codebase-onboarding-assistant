import React from 'react';

interface GuardrailBannerProps {
    message: string;
    type: 'warning' | 'error' | 'info';
}

const GuardrailBanner: React.FC<GuardrailBannerProps> = ({ message, type }) => {
    let bannerClass = '';

    switch (type) {
        case 'warning':
            bannerClass = 'guardrail-banner warning';
            break;
        case 'error':
            bannerClass = 'guardrail-banner error';
            break;
        case 'info':
            bannerClass = 'guardrail-banner info';
            break;
        default:
            bannerClass = 'guardrail-banner';
    }

    return (
        <div className={bannerClass}>
            {message}
        </div>
    );
};

export default GuardrailBanner;