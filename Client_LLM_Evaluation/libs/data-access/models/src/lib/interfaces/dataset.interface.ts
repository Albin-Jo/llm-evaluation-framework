export interface Dataset {
  id: string;
  name: string;
  description: string;
  documentCount: number;
  createdAt: string;
  updatedAt: string;
  type: string;
  tags?: string[];
  size?: number; // size in bytes
  format?: string;
  status: DatasetStatus;
  metadata?: Record<string, any>;
}

export enum DatasetStatus {
  PROCESSING = 'processing',
  READY = 'ready',
  ERROR = 'error'
}

export interface Document {
  id: string;
  datasetId: string;
  name: string;
  content: string;
  contentType?: string;
  metadata?: {
    size?: number;
    tokenCount?: number;
    format?: string;
    [key: string]: any;
  };
  createdAt: string;
  updatedAt?: string;
  embeddings?: boolean; // whether embeddings have been generated
}

export interface DatasetUploadRequest {
  name: string;
  description?: string;
  tags?: string[];
  files: File[] | FileList;
  type?: string; // Add type to upload request
}

export interface DatasetUpdateRequest {
  name?: string;
  description?: string;
  tags?: string[];
  type?: string; // Add type to update request
  [key: string]: any;
}

export interface DatasetListResponse {
  datasets: Dataset[];
  totalCount: number;
}

export interface DatasetDetailResponse {
  dataset: Dataset;
  documents: Document[];
}

// Add filter parameters interface
export interface DatasetFilterParams {
  page?: number;
  limit?: number;
  search?: string;
  status?: string;
  type?: string;
  tags?: string[];
  is_public?: boolean;
  sortBy?: 'name' | 'createdAt' | 'updatedAt' | 'documentCount';
  sortDirection?: 'asc' | 'desc';
  dateFrom?: string;
  dateTo?: string;
  sizeMin?: number;
  sizeMax?: number;
}
