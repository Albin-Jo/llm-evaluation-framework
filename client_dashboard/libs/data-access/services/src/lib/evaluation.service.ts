/* Path: libs/data-access/services/src/lib/evaluation.service.ts */
import { Injectable } from '@angular/core';
import { Observable, throwError } from 'rxjs';
import { catchError, map, tap } from 'rxjs/operators';
import { HttpParams, HttpErrorResponse } from '@angular/common/http';
import {
  Evaluation,
  EvaluationDetail,
  EvaluationFilterParams,
  EvaluationCreate,
  EvaluationUpdate,
  EvaluationProgress,
  EvaluationsResponse
} from '@ngtx-apps/data-access/models';
import { environment } from '@ngtx-apps/utils/shared';
import { HttpClientService } from './common/http-client.service';

@Injectable({
  providedIn: 'root'
})
export class EvaluationService {
  // Use the API endpoint path that matches FastAPI routes
  private baseUrl = '__fastapi__/evaluations';

  constructor(private httpClient: HttpClientService) {
    console.log('EvaluationService initialized with baseUrl:', this.baseUrl);
  }

  /**
   * Get a list of evaluations with optional filtering
   */
  getEvaluations(filters: EvaluationFilterParams = {}): Observable<EvaluationsResponse> {
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

    if (filters.status) {
      params = params.set('status', filters.status);
    }

    if (filters.agent_id) {
      params = params.set('agent_id', filters.agent_id);
    }

    if (filters.dataset_id) {
      params = params.set('dataset_id', filters.dataset_id);
    }

    console.log('Fetching evaluations with params:', params.toString());

    return this.httpClient.get<any>(this.baseUrl, params)
      .pipe(
        tap(response => console.log('Raw evaluation list response:', response)),
        map(response => {
          // Transform the API response to match the expected structure
          if (Array.isArray(response)) {
            console.log('Response is an array, transforming');
            return {
              evaluations: response,
              totalCount: response.length
            } as EvaluationsResponse;
          } else if (response.items && Array.isArray(response.items)) {
            // Handle paginated response format
            return {
              evaluations: response.items,
              totalCount: response.total || response.items.length
            } as EvaluationsResponse;
          }

          // Return the response as is if it already matches our format
          console.log('Returning response as is');
          return response as EvaluationsResponse;
        }),
        catchError(error => this.handleError('Failed to fetch evaluations', error))
      );
  }

  /**
   * Get a single evaluation by ID with detailed results
   */
  getEvaluation(id: string): Observable<EvaluationDetail> {
    console.log(`Fetching evaluation with ID: ${id}`);
    console.log(`Full URL being called: ${this.baseUrl}/${id}`);

    return this.httpClient.get<EvaluationDetail>(`${this.baseUrl}/${id}`)
      .pipe(
        tap(response => console.log('Raw evaluation detail response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to fetch evaluation with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to fetch evaluation with ID ${id}`, error);
        })
      );
  }

  /**
   * Create a new evaluation
   */
  createEvaluation(evaluation: EvaluationCreate): Observable<Evaluation> {
    console.log('Creating evaluation:', evaluation);

    return this.httpClient.post<Evaluation>(this.baseUrl, evaluation)
      .pipe(
        tap(response => console.log('Create evaluation response:', JSON.stringify(response))),
        catchError(error => {
          console.error('Failed to create evaluation', error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError('Failed to create evaluation', error);
        })
      );
  }

  /**
   * Update an existing evaluation
   */
  updateEvaluation(id: string, evaluation: EvaluationUpdate): Observable<Evaluation> {
    console.log(`Updating evaluation ${id}:`, JSON.stringify(evaluation));

    return this.httpClient.put<Evaluation>(`${this.baseUrl}/${id}`, evaluation)
      .pipe(
        tap(response => console.log('Update evaluation response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to update evaluation with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to update evaluation with ID ${id}`, error);
        })
      );
  }

  /**
   * Delete an evaluation
   */
  deleteEvaluation(id: string): Observable<void> {
    console.log(`Deleting evaluation with ID: ${id}`);
    console.log(`Delete URL: ${this.baseUrl}/${id}`);

    return this.httpClient.delete<void>(`${this.baseUrl}/${id}`)
      .pipe(
        tap(() => console.log(`Evaluation ${id} deleted successfully`)),
        catchError(error => {
          console.error(`Failed to delete evaluation with ID ${id}`, error);
          console.error('Delete error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to delete evaluation with ID ${id}`, error);
        })
      );
  }

  /**
   * Start an evaluation
   */
  startEvaluation(id: string): Observable<Evaluation> {
    console.log(`Starting evaluation with ID: ${id}`);
    console.log(`Start URL: ${this.baseUrl}/${id}/start`);

    return this.httpClient.post<Evaluation>(`${this.baseUrl}/${id}/start`, {})
      .pipe(
        tap(response => console.log('Start evaluation response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to start evaluation with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to start evaluation with ID ${id}`, error);
        })
      );
  }

  /**
   * Get evaluation progress
   */
  getEvaluationProgress(id: string): Observable<EvaluationProgress> {
    console.log(`Fetching progress for evaluation with ID: ${id}`);
    console.log(`Progress URL: ${this.baseUrl}/${id}/progress`);

    return this.httpClient.get<EvaluationProgress>(`${this.baseUrl}/${id}/progress`)
      .pipe(
        tap(response => console.log('Evaluation progress response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to fetch progress for evaluation with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to fetch progress for evaluation with ID ${id}`, error);
        })
      );
  }

  /**
   * Cancel an evaluation
   */
  cancelEvaluation(id: string): Observable<Evaluation> {
    console.log(`Cancelling evaluation with ID: ${id}`);
    console.log(`Cancel URL: ${this.baseUrl}/${id}/cancel`);

    return this.httpClient.post<Evaluation>(`${this.baseUrl}/${id}/cancel`, {})
      .pipe(
        tap(response => console.log('Cancel evaluation response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to cancel evaluation with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to cancel evaluation with ID ${id}`, error);
        })
      );
  }

  /**
   * Get evaluation results
   */
  getEvaluationResults(id: string): Observable<any[]> {
    console.log(`Fetching results for evaluation with ID: ${id}`);
    console.log(`Results URL: ${this.baseUrl}/${id}/results`);

    return this.httpClient.get<any[]>(`${this.baseUrl}/${id}/results`)
      .pipe(
        tap(response => console.log('Evaluation results response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to fetch results for evaluation with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to fetch results for evaluation with ID ${id}`, error);
        })
      );
  }

  /**
   * Test evaluation with sample data
   */
  testEvaluation(id: string, testData: any): Observable<any> {
    console.log(`Testing evaluation with ID: ${id}`);
    console.log(`Test input:`, JSON.stringify(testData));

    return this.httpClient.post<any>(`${this.baseUrl}/${id}/test`, testData)
      .pipe(
        tap(response => console.log('Test evaluation response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to test evaluation with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to test evaluation with ID ${id}`, error);
        })
      );
  }

  /**
   * Get supported metrics for a dataset type
   */
  getSupportedMetrics(datasetType: string): Observable<Record<string, string[]>> {
    console.log(`Fetching supported metrics for dataset type: ${datasetType}`);
    console.log(`Metrics URL: ${this.baseUrl}/metrics/${datasetType}`);

    return this.httpClient.get<Record<string, string[]>>(`${this.baseUrl}/metrics/${datasetType}`)
      .pipe(
        tap(response => console.log('Supported metrics response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to fetch supported metrics for dataset type ${datasetType}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to fetch supported metrics for dataset type ${datasetType}`, error);
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