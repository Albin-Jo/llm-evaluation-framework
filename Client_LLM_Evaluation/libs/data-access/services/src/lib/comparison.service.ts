import { Injectable } from '@angular/core';
import { Observable, throwError, of } from 'rxjs';
import { catchError, map, delay } from 'rxjs/operators';
import {
  HttpParams,
  HttpErrorResponse,
  HttpHeaders,
} from '@angular/common/http';
import {
  Comparison,
  ComparisonDetail,
  ComparisonFilterParams,
  ComparisonCreate,
  ComparisonUpdate,
  ComparisonsResponse,
  MetricDifference,
  VisualizationData,
} from '@ngtx-apps/data-access/models';
import { HttpClientService } from './common/http-client.service';
import { HttpClient } from '@angular/common/http';
import { AppConstant } from '@ngtx-apps/utils/shared';

@Injectable({
  providedIn: 'root',
})
export class ComparisonService {
  // Use the API endpoint path that matches FastAPI routes
  private baseUrl = '__fastapi__/comparisons';

  constructor(
    private httpClient: HttpClientService,
    private http: HttpClient // For blob downloads
  ) {}

  /**
   * Get a list of comparisons with optional filtering
   */
  getComparisons(
    filters: ComparisonFilterParams = {}
  ): Observable<ComparisonsResponse> {
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

    // Add evaluation_a_id filter
    if (filters.evaluation_a_id) {
      params = params.set('evaluation_a_id', filters.evaluation_a_id);
    }

    // Add evaluation_b_id filter
    if (filters.evaluation_b_id) {
      params = params.set('evaluation_b_id', filters.evaluation_b_id);
    }

    // Add name filter if provided
    if (filters.name) {
      params = params.set('name', filters.name);
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
            comparisons: response.items,
            totalCount: response.total || response.items.length,
          } as ComparisonsResponse;
        } else if (Array.isArray(response)) {
          // Fallback for backward compatibility
          return {
            comparisons: response,
            totalCount: response.length,
          } as ComparisonsResponse;
        }

        // Return the response as is if it already matches our format
        return response as ComparisonsResponse;
      }),
      catchError((error) =>
        this.handleError('Failed to fetch comparisons', error)
      )
    );
  }

  /**
   * Get a single comparison by ID with detailed results
   */
  getComparison(id: string): Observable<ComparisonDetail> {
    return this.httpClient.get<ComparisonDetail>(`${this.baseUrl}/${id}`).pipe(
      catchError((error) => {
        return this.handleError(
          `Failed to fetch comparison with ID ${id}`,
          error
        );
      })
    );
  }

  /**
   * Create a new comparison
   */
  createComparison(comparison: ComparisonCreate): Observable<Comparison> {
    return this.httpClient.post<Comparison>(this.baseUrl, comparison).pipe(
      catchError((error) => {
        return this.handleError('Failed to create comparison', error);
      })
    );
  }

  /**
   * Update an existing comparison
   */
  updateComparison(
    id: string,
    comparison: ComparisonUpdate
  ): Observable<Comparison> {
    return this.httpClient
      .put<Comparison>(`${this.baseUrl}/${id}`, comparison)
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to update comparison with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Delete a comparison
   */
  deleteComparison(id: string): Observable<void> {
    return this.httpClient.delete<void>(`${this.baseUrl}/${id}`).pipe(
      catchError((error) => {
        return this.handleError(
          `Failed to delete comparison with ID ${id}`,
          error
        );
      })
    );
  }

  /**
   * Run a comparison
   */
  runComparison(id: string): Observable<Comparison> {
    return this.httpClient
      .post<Comparison>(`${this.baseUrl}/${id}/run`, {})
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to run comparison with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Get comparison metrics
   */
  getComparisonMetrics(id: string): Observable<MetricDifference[]> {
    return this.httpClient
      .get<MetricDifference[]>(`${this.baseUrl}/${id}/metrics`)
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to fetch metrics for comparison with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Get visualization data
   */
  getVisualizationData(
    id: string,
    type: 'radar' | 'bar' | 'line'
  ): Observable<VisualizationData> {
    return this.httpClient
      .get<VisualizationData>(`${this.baseUrl}/${id}/visualizations/${type}`)
      .pipe(
        catchError((error) => {
          // If API endpoint not available yet, return mock data for development
          if (error.status === 404) {
            return this.getMockVisualizationData(type);
          }

          return this.handleError(
            `Failed to fetch ${type} visualization data for comparison with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Get comparison report
   */
  getComparisonReport(
    id: string,
    format: 'json' | 'html' | 'pdf' = 'json'
  ): Observable<any> {
    let params = new HttpParams().set('format', format);

    // For PDF format, we need to handle blob response
    if (format === 'pdf') {
      // Handle PDF downloads with appropriate response type using HttpClient
      // Construct the URL with the same pattern used in the interceptor
      const url = `${AppConstant.API_URL}/comparisons/${id}/report`;

      // Handle authentication manually for this special case
      let headers = new HttpHeaders();
      const token = localStorage.getItem('token');
      if (token) {
        headers = headers.set('Authorization', `Bearer ${token}`);
      }

      return this.http
        .get(url, {
          headers: headers,
          params: params,
          responseType: 'blob',
        })
        .pipe(
          catchError((error) => {
            return this.handleError(
              `Failed to download report for comparison with ID ${id}`,
              error
            );
          })
        );
    }

    return this.httpClient
      .get<any>(`${this.baseUrl}/${id}/report`, params)
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to fetch report for comparison with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Search comparisons with advanced criteria
   */
  searchComparisons(
    query?: string,
    filters?: Record<string, any>,
    page: number = 1,
    limit: number = 10,
    sortBy: string = 'created_at',
    sortDirection: 'asc' | 'desc' = 'desc'
  ): Observable<ComparisonsResponse> {
    const searchData = {
      query,
      filters,
      skip: (page - 1) * limit,
      limit,
      sort_by: sortBy,
      sort_dir: sortDirection,
    };

    return this.httpClient.post<any>(`${this.baseUrl}/search`, searchData).pipe(
      map((response) => {
        // Transform the API response to match the expected structure
        if (response && response.items && Array.isArray(response.items)) {
          return {
            comparisons: response.items,
            totalCount: response.total || response.items.length,
          } as ComparisonsResponse;
        }

        // Return the response as is if it already matches our format
        return response as ComparisonsResponse;
      }),
      catchError((error) =>
        this.handleError('Failed to search comparisons', error)
      )
    );
  }

  /**
   * Generate mock visualization data for development when API is not ready
   */
  private getMockVisualizationData(
    type: 'radar' | 'bar' | 'line'
  ): Observable<VisualizationData> {
    // Create mock data based on visualization type
    const metrics = [
      'Context Recall',
      'Context Precision',
      'Answer Similarity',
      'Answer Correctness',
      'Faithfulness',
      'Response Relevancy',
    ];

    let mockData: VisualizationData;

    if (type === 'radar') {
      mockData = {
        type: 'radar',
        labels: metrics,
        datasets: [
          {
            label: 'Evaluation A',
            data: [0.95, 0.82, 0.76, 0.68, 0.89, 0.71],
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderColor: 'rgba(54, 162, 235, 1)',
            fill: true,
          },
          {
            label: 'Evaluation B',
            data: [0.98, 0.87, 0.78, 0.72, 0.85, 0.81],
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgba(255, 99, 132, 1)',
            fill: true,
          },
        ],
      };
    } else if (type === 'bar') {
      mockData = {
        type: 'bar',
        labels: metrics,
        datasets: [
          {
            label: 'Evaluation A',
            data: [0.95, 0.82, 0.76, 0.68, 0.89, 0.71],
            backgroundColor: 'rgba(54, 162, 235, 0.7)',
          },
          {
            label: 'Evaluation B',
            data: [0.98, 0.87, 0.78, 0.72, 0.85, 0.81],
            backgroundColor: 'rgba(255, 99, 132, 0.7)',
          },
          {
            label: 'Difference (%)',
            data: [3.2, 6.1, 2.6, 5.9, -4.5, 14.1],
            backgroundColor: 'rgba(75, 192, 192, 0.7)',
          },
        ],
      };
    } else {
      // line chart
      mockData = {
        type: 'line',
        labels: metrics,
        datasets: [
          {
            label: 'Evaluation A',
            data: [0.95, 0.82, 0.76, 0.68, 0.89, 0.71],
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderColor: 'rgba(54, 162, 235, 1)',
            fill: false,
          },
          {
            label: 'Evaluation B',
            data: [0.98, 0.87, 0.78, 0.72, 0.85, 0.81],
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgba(255, 99, 132, 1)',
            fill: false,
          },
        ],
      };
    }

    // Return mock data with a small delay to simulate API call
    return of(mockData).pipe(delay(500));
  }

  /**
   * Handle errors from API calls
   */
  private handleError(message: string, error: any): Observable<never> {
    if (error instanceof HttpErrorResponse && error.status === 0) {
      // Network error or server not responding
      console.error('Network error or server not responding:', error);
    } else {
      // Other errors
      console.error(message, error);
    }

    return throwError(() => error);
  }
}
