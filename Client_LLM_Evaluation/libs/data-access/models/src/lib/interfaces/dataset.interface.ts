/* Path: libs/data-access/models/src/lib/interfaces/dataset.interface.ts */
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
}

export interface DatasetUpdateRequest {
  name?: string;
  description?: string;
  tags?: string[];
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

