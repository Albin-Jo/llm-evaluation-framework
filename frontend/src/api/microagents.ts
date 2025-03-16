import apiClient from './client';

export interface MicroAgent {
  id: string;
  name: string;
  description?: string;
  api_endpoint: string;
  domain: string;
  config?: any;
  is_active: boolean;
}

// Placeholder functions - these will be implemented in the future
export const getMicroAgents = async () => {
  // This is a placeholder function for future implementation
  console.warn('MicroAgents API not yet implemented');
  return { items: [], total: 0 };
};

export const getMicroAgent = async (id: string) => {
  // This is a placeholder function for future implementation
  console.warn('MicroAgents API not yet implemented');
  return {} as MicroAgent;
};

export const testMicroAgent = async (id: string, input: any) => {
  // This is a placeholder function for future implementation
  console.warn('MicroAgents API not yet implemented');
  return { output: 'Test response not available yet' };
};