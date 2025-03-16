import apiClient from './client';

export interface Evaluation {
  id: string;
  name: string;
  description?: string;
  method: 'ragas' | 'deepeval' | 'custom' | 'manual';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  config?: any;
  metrics: string[];
  experiment_id?: string;
  start_time?: string;
  end_time?: string;
  micro_agent_id: string;
  dataset_id: string;
  prompt_id: string;
}

// Placeholder functions - these will be implemented in the future
export const getEvaluations = async () => {
  // This is a placeholder function for future implementation
  console.warn('Evaluations API not yet implemented');
  return { items: [], total: 0 };
};

export const getEvaluation = async (id: string) => {
  // This is a placeholder function for future implementation
  console.warn('Evaluations API not yet implemented');
  return {} as Evaluation;
};

export const createEvaluation = async (data: any) => {
  // This is a placeholder function for future implementation
  console.warn('Evaluations API not yet implemented');
  return {} as Evaluation;
};

export const updateEvaluation = async (id: string, data: any) => {
  // This is a placeholder function for future implementation
  console.warn('Evaluations API not yet implemented');
  return {} as Evaluation;
};

export const deleteEvaluation = async (id: string) => {
  // This is a placeholder function for future implementation
  console.warn('Evaluations API not yet implemented');
  return {};
};

export const startEvaluation = async (id: string) => {
  // This is a placeholder function for future implementation
  console.warn('Evaluations API not yet implemented');
  return {};
};

export const cancelEvaluation = async (id: string) => {
  // This is a placeholder function for future implementation
  console.warn('Evaluations API not yet implemented');
  return {};
};

export const getEvaluationResults = async (id: string) => {
  // This is a placeholder function for future implementation
  console.warn('Evaluations API not yet implemented');
  return {};
};