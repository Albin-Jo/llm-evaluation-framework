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
  EvaluationsResponse,
} from '@ngtx-apps/data-access/models';
import { environment } from '@ngtx-apps/utils/shared';
import { HttpClientService } from './common/http-client.service';

@Injectable({
  providedIn: 'root',
})
export class EvaluationService {
  // Use the API endpoint path that matches FastAPI routes
  private baseUrl = '__fastapi__/evaluations';

  constructor(private httpClient: HttpClientService) {}

  /**
   * Get a list of evaluations with optional filtering
   */
  getEvaluations(
    filters: EvaluationFilterParams = {}
  ): Observable<EvaluationsResponse> {
    // Convert the filters object to HttpParams
    let params = new HttpParams();

    // Add each parameter if it exists
    // Convert page to skip parameter (for pagination)
    if (filters.page !== undefined) {
      params = params.set(
        'skip',
        ((filters.page - 1) * (filters.limit || 10)).toString()
      );
    } else {
      params = params.set('skip', '0');
    }

    // Add limit parameter for pagination
    if (filters.limit !== undefined) {
      params = params.set('limit', filters.limit.toString());
    } else {
      params = params.set('limit', '10'); // Default limit
    }

    // Add status filter
    if (filters.status) {
      params = params.set('status', filters.status);
    }

    // Add agent_id filter
    if (filters.agent_id) {
      params = params.set('agent_id', filters.agent_id);
    }

    // Add dataset_id filter
    if (filters.dataset_id) {
      params = params.set('dataset_id', filters.dataset_id);
    }

    // Add name filter if provided
    if (filters.name) {
      params = params.set('name', filters.name);
    }

    // Add method filter if provided
    if (filters.method) {
      params = params.set('method', filters.method);
    }

    // Add sort parameters - always sending these parameters to match backend expectations
    // Default to created_at and desc if not provided
    params = params.set('sort_by', filters.sortBy || 'created_at');
    params = params.set('sort_dir', filters.sortDirection || 'desc');

    return this.httpClient.get<any>(this.baseUrl, params).pipe(
      map((response) => {
        // Transform the API response to match the expected structure
        if (response && response.items && Array.isArray(response.items)) {
          // The updated API now returns {items: [...], total: number}
          return {
            evaluations: response.items,
            totalCount: response.total || response.items.length,
          } as EvaluationsResponse;
        } else if (Array.isArray(response)) {
          // Fallback for backward compatibility
          return {
            evaluations: response,
            totalCount: response.length,
          } as EvaluationsResponse;
        }

        // Return the response as is if it already matches our format
        return response as EvaluationsResponse;
      }),
      catchError((error) =>
        this.handleError('Failed to fetch evaluations', error)
      )
    );
  }

  /**
   * Get a single evaluation by ID with detailed results
   */
  getEvaluation(id: string): Observable<EvaluationDetail> {
    return this.httpClient.get<EvaluationDetail>(`${this.baseUrl}/${id}`).pipe(
      catchError((error) => {
        return this.handleError(
          `Failed to fetch evaluation with ID ${id}`,
          error
        );
      })
    );
  }

  /**
   * Create a new evaluation
   */
  createEvaluation(evaluation: EvaluationCreate): Observable<Evaluation> {
    return this.httpClient.post<Evaluation>(this.baseUrl, evaluation).pipe(
      catchError((error) => {
        return this.handleError('Failed to create evaluation', error);
      })
    );
  }

  /**
   * Update an existing evaluation
   */
  updateEvaluation(
    id: string,
    evaluation: EvaluationUpdate
  ): Observable<Evaluation> {
    return this.httpClient
      .put<Evaluation>(`${this.baseUrl}/${id}`, evaluation)
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to update evaluation with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Delete an evaluation
   */
  deleteEvaluation(id: string): Observable<void> {
    return this.httpClient.delete<void>(`${this.baseUrl}/${id}`).pipe(
      catchError((error) => {
        return this.handleError(
          `Failed to delete evaluation with ID ${id}`,
          error
        );
      })
    );
  }

  /**
   * Start an evaluation
   */
  startEvaluation(id: string): Observable<Evaluation> {
    return this.httpClient
      .post<Evaluation>(`${this.baseUrl}/${id}/start`, {})
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to start evaluation with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Get evaluation progress
   */
  getEvaluationProgress(id: string): Observable<EvaluationProgress> {
    return this.httpClient
      .get<EvaluationProgress>(`${this.baseUrl}/${id}/progress`)
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to fetch progress for evaluation with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Cancel an evaluation
   */
  cancelEvaluation(id: string): Observable<Evaluation> {
    return this.httpClient
      .post<Evaluation>(`${this.baseUrl}/${id}/cancel`, {})
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to cancel evaluation with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Get evaluation results
   */
  getEvaluationResults(id: string, skip: number = 0, limit: number = 100) {
    let params = new HttpParams()
      .set('skip', skip.toString())
      .set('limit', limit.toString());
    return this.httpClient
      .get<any>(`${this.baseUrl}/${id}/results`, params)
      .pipe(
        map((response) => {
          // Handle different response formats
          if (response && response.items && Array.isArray(response.items)) {
            // Backend returns {items: [...], total: number}
            return {
              items: response.items,
              total: response.total || response.items.length,
            };
          } else if (Array.isArray(response)) {
            // Backend returns array directly
            return {
              items: response,
              total: response.length,
            };
          } else {
            // Fallback to empty response
            return {
              items: [],
              total: 0,
            };
          }
        }),
        catchError((error) => {
          return this.handleError(
            `Failed to fetch results for evaluation with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Test evaluation with sample data
   */
  testEvaluation(id: string, testData: any): Observable<any> {
    return this.httpClient
      .post<any>(`${this.baseUrl}/${id}/test`, testData)
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to test evaluation with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Get supported metrics for a dataset type and evaluation method
   */
  getSupportedMetricsByMethod(
    datasetType: string,
    evaluationMethod: string
  ): Observable<Record<string, any>> {
    return this.httpClient
      .get<Record<string, any>>(
        `${this.baseUrl}/metrics/${datasetType}/${evaluationMethod}`
      )
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to fetch supported metrics for dataset type ${datasetType} and method ${evaluationMethod}`,
            error
          );
        })
      );
  }

  /**
   * Get supported metrics for a dataset type
   */
  getSupportedMetrics(
    datasetType: string,
    evaluationMethod?: string
  ): Observable<Record<string, any>> {
    // If evaluation method is provided, use the new endpoint
    if (evaluationMethod) {
      return this.getSupportedMetricsByMethod(datasetType, evaluationMethod);
    }

    // Otherwise, use the old endpoint (defaults to RAGAS)
    return this.httpClient
      .get<Record<string, any>>(`${this.baseUrl}/metrics/${datasetType}`)
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to fetch supported metrics for dataset type ${datasetType}`,
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
