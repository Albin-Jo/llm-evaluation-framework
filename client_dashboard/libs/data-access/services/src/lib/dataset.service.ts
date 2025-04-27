/* Path: libs/data-access/services/src/lib/dataset.service.ts */
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
  Document
} from '@ngtx-apps/data-access/models';
import { environment } from '@ngtx-apps/utils/shared';
import { HttpClientService } from './common/http-client.service';

@Injectable({
  providedIn: 'root'
})
export class DatasetService {
  // Use the API endpoint path that matches your FastAPI routes
  private baseUrl = '__fastapi__/datasets';

  constructor(private httpClient: HttpClientService) {
    console.log('DatasetService initialized with baseUrl:', this.baseUrl);
  }

/**
 * Get a list of datasets with optional filtering
 */
getDatasets(filters: DatasetFilterParams = {}): Observable<DatasetListResponse> {
  // Convert the filters object to HttpParams
  let params = new HttpParams();

  // Add each parameter if it exists
  if (filters.page !== undefined) {
    params = params.set('skip', ((filters.page - 1) * (filters.limit || 10)).toString());
  } else {
    params = params.set('skip', '0');
  }

  if (filters.limit !== undefined) {
    params = params.set('limit', filters.limit.toString());
  } else {
    params = params.set('limit', '100');
  }

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

  console.log('Fetching datasets with params:', params.toString());

  return this.httpClient.get<any>(this.baseUrl, params)
    .pipe(
      tap(response => console.log('Raw dataset list response:', response)),
      map(response => {
        // Transform the API response to match the expected structure
        if (Array.isArray(response)) {
          console.log('Response is an array, transforming');
          return {
            datasets: response.map(item => this.transformDataset(item)),
            totalCount: response.length
          } as DatasetListResponse;
        } else if (response.items && Array.isArray(response.items)) {
          // Handle paginated response format
          return {
            datasets: response.items.map((item: any) => this.transformDataset(item)),
            totalCount: response.total || response.items.length
          } as DatasetListResponse;
        }

        // Return the response as is if it already matches our format
        console.log('Returning response as is');
        return response as DatasetListResponse;
      }),
      catchError(error => this.handleError('Failed to fetch datasets', error))
    );
}

  /**
   * Get a single dataset by ID
   */
  getDataset(id: string): Observable<DatasetDetailResponse> {
    console.log(`Fetching dataset with ID: ${id}`);
    console.log(`Full URL being called: ${this.baseUrl}/${id}`);

    return this.httpClient.get<any>(`${this.baseUrl}/${id}`)
      .pipe(
        tap(response => console.log('Raw dataset detail response:', JSON.stringify(response))),
        map(response => {
          console.log('Transforming dataset detail response');

          // If the response is already in the expected format, return it
          if (response.dataset) {
            return response as DatasetDetailResponse;
          }

          // Transform the response to match our expected structure
          const detailResponse: DatasetDetailResponse = {
            dataset: this.transformDataset(response),
            documents: response.documents || []
          };

          console.log('Transformed dataset detail response:', JSON.stringify(detailResponse));
          return detailResponse;
        }),
        catchError(error => {
          console.error(`Failed to fetch dataset with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to fetch dataset with ID ${id}`, error);
        })
      );
  }

  private transformDataset(data: any): Dataset {
    console.log('Transforming dataset data:', JSON.stringify(data));

    // Transform API response to match our Dataset interface
    const transformed: Dataset = {
      id: data.id || '',
      name: data.name || '',
      description: data.description || '',
      documentCount: data.row_count || 0,
      createdAt: data.created_at || new Date().toISOString(),
      updatedAt: data.updated_at || new Date().toISOString(),
      tags: data.tags || [],
      size: data.size || 0,
      status: data.status || 'ready',
      type: data.type || ''
    };

    console.log('Transformed dataset:', JSON.stringify(transformed));
    return transformed;
  }

  /**
   * Upload a new dataset with files
   */
  uploadDataset(request: DatasetUploadRequest): Observable<Dataset> {
    console.log('Uploading dataset:', request.name);
    console.log('File count:', request.files ? request.files.length : 0);

    const formData = new FormData();
    formData.append('name', request.name);

    if (request.description) {
      formData.append('description', request.description);
    }

    if (request.tags && request.tags.length > 0) {
      // For FastAPI, we need to handle tags as a string
      formData.append('tags', JSON.stringify(request.tags));
    }

    // Set default dataset type to match API expectation
    formData.append('type', 'user_query');
    formData.append('is_public', 'true');

    // Append the file to the form data
    if (request.files && request.files.length > 0) {
      formData.append('file', request.files[0]);
      console.log('File attached:', request.files[0].name);
    }

    // Log FormData entries (can't directly console.log the FormData)
    formData.forEach((value, key) => {
      console.log(`FormData contains: ${key} = ${value instanceof File ? value.name : value}`);
    });

    // Using the correct parameter order according to your HttpClientService interface
    return this.httpClient.post<any>(this.baseUrl, formData)
      .pipe(
        tap(response => console.log('Upload response:', JSON.stringify(response))),
        map(response => this.transformDataset(response)),
        catchError(error => {
          console.error('Upload error:', error);
          console.error('Upload error details:', JSON.stringify(error, null, 2));
          return this.handleError('Failed to upload dataset', error);
        })
      );
  }

  /**
   * Upload documents to an existing dataset
   * @param datasetId ID of the dataset to upload documents to
   * @param files Files to upload
   */
  uploadDocumentsToDataset(datasetId: string, files: File[]): Observable<Dataset> {
    console.log(`Uploading documents to dataset ${datasetId}`);
    console.log('File count:', files ? files.length : 0);

    const formData = new FormData();

    // Append each file to the form data
    if (files && files.length > 0) {
      files.forEach((file, index) => {
        formData.append('file', file, file.name);
      });
      console.log(`Attached ${files.length} files to the request`);
    }

    // Using the correct endpoint for adding documents to an existing dataset
    return this.httpClient.post<any>(`${this.baseUrl}/${datasetId}/documents`, formData)
      .pipe(
        tap(response => console.log('Document upload response:', JSON.stringify(response))),
        map(response => this.transformDataset(response)),
        catchError(error => {
          console.error(`Failed to upload documents to dataset ${datasetId}`, error);
          console.error('Document upload error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to upload documents to dataset ${datasetId}`, error);
        })
      );
  }

  /**
   * Update an existing dataset
   */
  updateDataset(id: string, request: DatasetUpdateRequest): Observable<Dataset> {
    console.log(`Updating dataset ${id}:`, JSON.stringify(request));

    // Transform our request to match the FastAPI expected structure
    const apiRequest = {
      name: request.name,
      description: request.description,
      type: "user_query",
      schema_definition: {},
      meta_info: {},
      version: "1.0",
      row_count: 0,
      is_public: true
    };

    console.log('Transformed update request:', JSON.stringify(apiRequest));

    return this.httpClient.put<any>(`${this.baseUrl}/${id}`, apiRequest)
      .pipe(
        tap(response => console.log('Update response:', JSON.stringify(response))),
        map(response => this.transformDataset(response)),
        catchError(error => {
          console.error(`Failed to update dataset with ID ${id}`, error);
          console.error('Update error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to update dataset with ID ${id}`, error);
        })
      );
  }

  /**
   * Delete a dataset
   */
  deleteDataset(id: string): Observable<void> {
    console.log(`Deleting dataset with ID: ${id}`);
    console.log(`Delete URL: ${this.baseUrl}/${id}`);

    return this.httpClient.delete<void>(`${this.baseUrl}/${id}`)
      .pipe(
        tap(() => console.log(`Dataset ${id} deleted successfully`)),
        catchError(error => {
          console.error(`Failed to delete dataset with ID ${id}`, error);
          console.error('Delete error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to delete dataset with ID ${id}`, error);
        })
      );
  }

  /**
   * Get documents for a dataset
   */
  getDocuments(datasetId: string, page = 1, limit = 20): Observable<Document[]> {
    console.log(`Fetching documents for dataset ${datasetId}, page ${page}, limit ${limit}`);

    const params = new HttpParams()
      .set('page', page.toString())
      .set('limit', limit.toString());

    return this.httpClient.get<Document[]>(`${this.baseUrl}/${datasetId}/documents`, params)
      .pipe(
        tap(response => console.log('Documents response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to fetch documents for dataset with ID ${datasetId}`, error);
          console.error('Documents error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to fetch documents for dataset with ID ${datasetId}`, error);
        })
      );
  }

  /**
   * Handle errors from API calls
   */
  private handleError(message: string, error: any): Observable<never> {
    console.error(message, error);

    // Check if error is status 0, which usually indicates network/CORS issues
    if (error instanceof HttpErrorResponse && error.status === 0) {
      console.error('Network Error - possible CORS issue or server unavailable');
      console.error('Attempted URL:', error.url);
    }

    // Log detailed error information
    console.error('Error status:', error.status);
    console.error('Error message:', error.message);
    if (error.error) {
      console.error('Server error response:', error.error);
    }

    return throwError(() => error);
  }
}
