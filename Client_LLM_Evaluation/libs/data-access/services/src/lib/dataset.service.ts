import { Injectable } from '@angular/core';
import { Observable, throwError } from 'rxjs';
import { catchError, map, tap } from 'rxjs/operators';
import { HttpParams, HttpErrorResponse } from '@angular/common/http';
import {
  Dataset,
  DatasetFilterParams,
  DatasetListResponse,
  DatasetUploadRequest,
  DatasetUpdateRequest,
  DatasetDetailResponse,
  Document,
} from '@ngtx-apps/data-access/models';
import { environment } from '@ngtx-apps/utils/shared';
import { HttpClientService } from './common/http-client.service';

@Injectable({
  providedIn: 'root',
})
export class DatasetService {
  private baseUrl = '__fastapi__/datasets';

  constructor(private httpClient: HttpClientService) {}

  /**
   * Get a list of datasets with optional filtering
   */
  getDatasets(
    filters: DatasetFilterParams = {}
  ): Observable<DatasetListResponse> {
    let params = new HttpParams();

    // Handle pagination parameters
    if (filters.page !== undefined) {
      const skip = (filters.page - 1) * (filters.limit || 10);
      params = params.set('skip', skip.toString());
    } else {
      params = params.set('skip', '0');
    }

    if (filters.limit !== undefined) {
      params = params.set('limit', filters.limit.toString());
    } else {
      params = params.set('limit', '100');
    }

    // Add other filters
    if (filters.search) {
      params = params.set('search', filters.search);
    }

    if (filters.status) {
      params = params.set('status', filters.status);
    }

    if (filters.type) {
      params = params.set('type', filters.type);
    }

    // Handle is_public parameter
    const isPublic = filters.is_public !== undefined ? filters.is_public : true;
    params = params.set('is_public', isPublic.toString());

    // Add date filtering if provided
    if (filters.dateFrom) {
      params = params.set('date_from', filters.dateFrom);
    }

    if (filters.dateTo) {
      params = params.set('date_to', filters.dateTo);
    }

    // Add size filtering if provided
    if (filters.sizeMin !== undefined) {
      params = params.set('size_min', filters.sizeMin.toString());
    }

    if (filters.sizeMax !== undefined) {
      params = params.set('size_max', filters.sizeMax.toString());
    }

    // Add sorting parameters
    if (filters.sortBy) {
      params = params.set('sort_by', filters.sortBy);
    }

    if (filters.sortDirection) {
      params = params.set('sort_direction', filters.sortDirection);
    }

    // Add tags filtering if provided
    if (filters.tags && filters.tags.length > 0) {
      params = params.set('tags', filters.tags.join(','));
    }

    return this.httpClient.get<any>(this.baseUrl, params).pipe(
      map((response) => {
        // The API returns a paginated response structure
        if (response.items && Array.isArray(response.items)) {
          return {
            datasets: response.items.map((item: any) =>
              this.transformDataset(item)
            ),
            totalCount: response.total || response.items.length,
          } as DatasetListResponse;
        }

        // Fallback for unexpected response format
        if (Array.isArray(response)) {
          return {
            datasets: response.map((item) => this.transformDataset(item)),
            totalCount: response.length,
          } as DatasetListResponse;
        }

        return {
          datasets: [],
          totalCount: 0,
        } as DatasetListResponse;
      }),
      catchError((error) => this.handleError('Failed to fetch datasets', error))
    );
  }

  /**
   * Get a single dataset by ID
   */
  getDataset(id: string): Observable<DatasetDetailResponse> {
    return this.httpClient.get<any>(`${this.baseUrl}/${id}`).pipe(
      map((response) => {
        // Create a DatasetDetailResponse
        const detailResponse: DatasetDetailResponse = {
          dataset: this.transformDataset(response),
          documents: [],
        };

        // If the dataset has rows and file information, create a document entry
        if (response.row_count > 0 && response.meta_info?.filename) {
          const document: Document = {
            id: response.id,
            datasetId: response.id,
            name: response.meta_info.filename,
            content: '',
            contentType: response.meta_info.content_type,
            metadata: {
              size: response.meta_info.size,
              format: response.meta_info.content_type?.split('/')[1] || 'json',
              row_count: response.meta_info.row_count,
            },
            createdAt: response.created_at,
            updatedAt: response.updated_at,
          };
          detailResponse.documents = [document];
        }

        return detailResponse;
      }),
      catchError((error) => {
        return this.handleError(`Failed to fetch dataset with ID ${id}`, error);
      })
    );
  }

  /**
   * Transform API dataset to our Dataset interface
   */
  private transformDataset(data: any): Dataset {
    const transformed: Dataset = {
      id: data.id || '',
      name: data.name || '',
      description: data.description || '',
      documentCount: data.row_count || 0,
      createdAt: data.created_at || new Date().toISOString(),
      updatedAt: data.updated_at || new Date().toISOString(),
      tags: data.tags || [],
      size: data.meta_info?.size || 0,
      status: data.status || 'ready',
      type: data.type || '',
      metadata: {
        file_path: data.file_path,
        version: data.version,
        is_public: data.is_public,
        schema_definition: data.schema_definition,
        meta_info: data.meta_info,
      },
    };

    return transformed;
  }

  /**
   * Upload a new dataset with files
   */
  uploadDataset(request: DatasetUploadRequest): Observable<Dataset> {
    const formData = new FormData();
    formData.append('name', request.name);

    if (request.description) {
      formData.append('description', request.description);
    }

    if (request.tags && request.tags.length > 0) {
      formData.append('tags', JSON.stringify(request.tags));
    }

    // Set default dataset type if not provided
    const datasetType = request.type || 'user_query';
    formData.append('type', datasetType);
    formData.append('is_public', 'true');

    // Append the file to the form data
    if (request.files && request.files.length > 0) {
      formData.append('file', request.files[0]);
    }

    return this.httpClient.post<any>(this.baseUrl, formData).pipe(
      map((response) => this.transformDataset(response)),
      catchError((error) => {
        return this.handleError('Failed to upload dataset', error);
      })
    );
  }

  /**
   * Upload documents to an existing dataset
   */
  uploadDocumentsToDataset(
    datasetId: string,
    files: File[]
  ): Observable<Dataset> {
    const formData = new FormData();

    // Append each file to the form data
    if (files && files.length > 0) {
      files.forEach((file, index) => {
        formData.append('file', file, file.name);
      });
    }

    return this.httpClient
      .post<any>(`${this.baseUrl}/${datasetId}/documents`, formData)
      .pipe(
        map((response) => this.transformDataset(response)),
        catchError((error) => {
          return this.handleError(
            `Failed to upload documents to dataset ${datasetId}`,
            error
          );
        })
      );
  }

  /**
   * Update an existing dataset
   */
  updateDataset(
    id: string,
    request: DatasetUpdateRequest
  ): Observable<Dataset> {
    // Transform our request to match the FastAPI expected structure
    const apiRequest = {
      name: request.name,
      description: request.description,
      type: request.type || 'user_query',
      schema_definition: {},
      meta_info: {},
      version: '1.0',
      row_count: 0,
      is_public: true,
    };

    return this.httpClient.put<any>(`${this.baseUrl}/${id}`, apiRequest).pipe(
      map((response) => this.transformDataset(response)),
      catchError((error) => {
        return this.handleError(
          `Failed to update dataset with ID ${id}`,
          error
        );
      })
    );
  }

  /**
   * Delete a dataset
   */
  deleteDataset(id: string): Observable<void> {
    return this.httpClient.delete<void>(`${this.baseUrl}/${id}`).pipe(
      catchError((error) => {
        return this.handleError(
          `Failed to delete dataset with ID ${id}`,
          error
        );
      })
    );
  }

  /**
   * Get documents for a dataset
   */
  getDocuments(
    datasetId: string,
    page = 1,
    limit = 20
  ): Observable<Document[]> {
    const params = new HttpParams()
      .set('page', page.toString())
      .set('limit', limit.toString());

    return this.httpClient
      .get<Document[]>(`${this.baseUrl}/${datasetId}/documents`, params)
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to fetch documents for dataset with ID ${datasetId}`,
            error
          );
        })
      );
  }

  /**
   * Handle errors from API calls
   */
  private handleError(message: string, error: any): Observable<never> {
    if (error instanceof HttpErrorResponse && error.status === 0) {
    }

    return throwError(() => error);
  }
}
