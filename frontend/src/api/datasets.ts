import apiClient from './client';

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  type: 'user_query' | 'context' | 'question_answer' | 'conversation' | 'custom';
  file_path: string;
  schema?: any;
  meta_info?: any;
  version: string;
  row_count?: number;
  is_public: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface DatasetPreview {
  headers: string[];
  rows: any[][];
  total_rows: number;
}

export interface CreateDatasetRequest {
  name: string;
  description?: string;
  type: Dataset['type'];
  file: File;
  is_public?: boolean;
}

export interface UpdateDatasetRequest {
  name?: string;
  description?: string;
  type?: Dataset['type'];
  is_public?: boolean;
}

// Get all datasets with optional pagination and filters
export const getDatasets = async (params?: {
  page?: number;
  limit?: number;
  search?: string;
  type?: Dataset['type'];
}) => {
  const response = await apiClient.get('/api/datasets/', { params });
  return response.data;
};

// Get a single dataset by ID
export const getDataset = async (id: string) => {
  const response = await apiClient.get(`/api/datasets/${id}`);
  return response.data as Dataset;
};

// Create a new dataset
export const createDataset = async (data: CreateDatasetRequest) => {
  const formData = new FormData();
  formData.append('name', data.name);
  if (data.description) formData.append('description', data.description);
  formData.append('type', data.type);
  formData.append('file', data.file);
  if (data.is_public !== undefined) formData.append('is_public', String(data.is_public));

  const response = await apiClient.post('/api/datasets/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data as Dataset;
};

// Update an existing dataset
export const updateDataset = async (id: string, data: UpdateDatasetRequest) => {
  const response = await apiClient.put(`/api/datasets/${id}`, data);
  return response.data as Dataset;
};

// Delete a dataset
export const deleteDataset = async (id: string) => {
  const response = await apiClient.delete(`/api/datasets/${id}`);
  return response.data;
};

// Get a preview of dataset content
export const getDatasetPreview = async (id: string, params?: { limit?: number }) => {
  const response = await apiClient.get(`/api/datasets/${id}/preview`, { params });
  return response.data as DatasetPreview;
};