
import React, { createContext, useContext } from 'react';
import { SDKContextType } from './types';

export const SDKContext = createContext<SDKContextType | null>(null);

export const useSDK = () => {
    const context = useContext(SDKContext);
    if (!context) throw new Error("useSDK must be used within SDKProvider");
    return context;
};
