import apiClient from './client';

export interface Comparison {
  id: string;
  name: string;
  description?: string;
  evaluation_a_id: string;
  evaluation_b_id: string;
  comparison_results?: any;
}

// Placeholder functions - these will be implemented in the future
export const getComparisons = async () => {
  // This is a placeholder function for future implementation
  console.warn('Comparisons API not yet implemented');
  return { items: [], total: 0 };
};

export const getComparison = async (id: string) => {
  // This is a placeholder function for future implementation
  console.warn('Comparisons API not yet implemented');
  return {} as Comparison;
};

export const createComparison = async (data: any) => {
  // This is a placeholder function for future implementation
  console.warn('Comparisons API not yet implemented');
  return {} as Comparison;
};

export const deleteComparison = async (id: string) => {
  // This is a placeholder function for future implementation
  console.warn('Comparisons API not yet implemented');
  return {};
};