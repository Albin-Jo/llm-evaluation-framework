/* Path: libs/data-access/models/src/lib/interfaces/dataset-filter.interface.ts */
export interface DatasetFilterParams {
  page?: number;
  limit?: number;
  search?: string;
  status?: string;
  type?: string;
  is_public?: boolean;
  sortBy?: 'name' | 'createdAt' | 'updatedAt' | 'documentCount';
  sortDirection?: 'asc' | 'desc';
  tags?: string[];
  dateFrom?: string;
  dateTo?: string;
  sizeMin?: number;
  sizeMax?: number;
}
