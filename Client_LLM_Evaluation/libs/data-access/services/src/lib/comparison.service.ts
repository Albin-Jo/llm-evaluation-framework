import { Injectable } from '@angular/core';
import { Observable, throwError, of, forkJoin, combineLatest } from 'rxjs';
import { catchError, map, delay, tap, switchMap } from 'rxjs/operators';
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
  ApiVisualizationData,
} from '@ngtx-apps/data-access/models';
import { HttpClientService } from './common/http-client.service';
import { HttpClient } from '@angular/common/http';
import { AppConstant } from '@ngtx-apps/utils/shared';

@Injectable({
  providedIn: 'root',
})
export class ComparisonService {
  private baseUrl = '__fastapi__/comparisons';
  private datasetsUrl = '__fastapi__/datasets';
  private agentsUrl = '__fastapi__/agents';
  private promptsUrl = '__fastapi__/prompts';
  private evaluationsUrl = '__fastapi__/evaluations';

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
      params = params.set('evaluation_id', filters.evaluation_a_id);
    }

    // Add evaluation_b_id filter - only if evaluation_a_id is not set
    else if (filters.evaluation_b_id) {
      params = params.set('evaluation_id', filters.evaluation_b_id);
    }

    // Add name filter if provided
    if (filters.name) {
      params = params.set('search', filters.name);
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

        return response as ComparisonsResponse;
      }),
      catchError((error) => {
        return this.handleError('Failed to fetch comparisons', error);
      })
    );
  }

  /**
   * Get a single comparison by ID with detailed results and enhanced metadata
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
   * Enhance comparison with metadata (dataset, agent, prompt names)
   */
  private enhanceComparisonWithMetadata(
    comparison: ComparisonDetail
  ): Observable<ComparisonDetail> {
    // Create array of metadata requests
    const metadataRequests: Observable<any>[] = [];

    // Get evaluation A metadata if available
    if (comparison.evaluation_a) {
      metadataRequests.push(
        this.getEvaluationMetadata(comparison.evaluation_a)
      );
    } else {
      metadataRequests.push(of(null));
    }

    // Get evaluation B metadata if available
    if (comparison.evaluation_b) {
      metadataRequests.push(
        this.getEvaluationMetadata(comparison.evaluation_b)
      );
    } else {
      metadataRequests.push(of(null));
    }

    if (!comparison.evaluation_a || !comparison.evaluation_b) {
      return this.fetchEvaluationData(comparison).pipe(
        switchMap((enhancedComparison) =>
          this.enhanceComparisonWithMetadata(enhancedComparison)
        )
      );
    }

    return combineLatest(metadataRequests).pipe(
      map(([evaluationAMeta, evaluationBMeta]) => {
        // Enhanced comparison with metadata
        const enhanced = { ...comparison };

        if (evaluationAMeta && enhanced.evaluation_a) {
          enhanced.evaluation_a = {
            ...enhanced.evaluation_a,
            ...evaluationAMeta,
          };
        }

        if (evaluationBMeta && enhanced.evaluation_b) {
          enhanced.evaluation_b = {
            ...enhanced.evaluation_b,
            ...evaluationBMeta,
          };
        }

        return enhanced;
      }),
      catchError((error) => {
        console.warn('Failed to enhance comparison with metadata:', error);
        // Return original comparison if metadata fetch fails
        return of(comparison);
      })
    );
  }

  /**
   * Get evaluation metadata (dataset, agent, prompt names)
   */
  private getEvaluationMetadata(evaluation: any): Observable<any> {
    const metadataRequests: { [key: string]: Observable<any> } = {};

    // Get dataset name
    if (evaluation.dataset_id) {
      metadataRequests['dataset'] = this.getDatasetName(evaluation.dataset_id);
    }

    // Get agent name
    if (evaluation.agent_id) {
      metadataRequests['agent'] = this.getAgentName(evaluation.agent_id);
    }

    // Get prompt name
    if (evaluation.prompt_id) {
      metadataRequests['prompt'] = this.getPromptName(evaluation.prompt_id);
    }

    if (Object.keys(metadataRequests).length === 0) {
      return of({
        dataset_name: 'Unknown Dataset',
        agent_name: 'Unknown Agent',
        prompt_name: 'Default Prompt',
      });
    }

    return forkJoin(metadataRequests).pipe(
      map((results) => ({
        dataset_name: results['dataset'] || 'Unknown Dataset',
        agent_name: results['agent'] || 'Unknown Agent',
        prompt_name: results['prompt'] || 'Default Prompt',
      })),
      catchError(() =>
        of({
          dataset_name: 'Unknown Dataset',
          agent_name: 'Unknown Agent',
          prompt_name: 'Default Prompt',
        })
      )
    );
  }

  /**
   * Fetch evaluation data if not present in comparison
   */
  private fetchEvaluationData(
    comparison: ComparisonDetail
  ): Observable<ComparisonDetail> {
    const evaluationRequests: Observable<any>[] = [];

    // Fetch evaluation A if missing
    if (!comparison.evaluation_a && comparison.evaluation_a_id) {
      evaluationRequests.push(
        this.getEvaluationById(comparison.evaluation_a_id)
      );
    } else {
      evaluationRequests.push(of(comparison.evaluation_a));
    }

    // Fetch evaluation B if missing
    if (!comparison.evaluation_b && comparison.evaluation_b_id) {
      evaluationRequests.push(
        this.getEvaluationById(comparison.evaluation_b_id)
      );
    } else {
      evaluationRequests.push(of(comparison.evaluation_b));
    }

    return combineLatest(evaluationRequests).pipe(
      map(([evaluationA, evaluationB]) => ({
        ...comparison,
        evaluation_a: evaluationA,
        evaluation_b: evaluationB,
      }))
    );
  }

  /**
   * Get evaluation by ID
   */
  private getEvaluationById(id: string): Observable<any> {
    return this.httpClient
      .get<any>(`${this.evaluationsUrl}/${id}`)
      .pipe(catchError(() => of(null)));
  }

  /**
   * Get dataset name by ID
   */
  private getDatasetName(id: string): Observable<string> {
    return this.httpClient.get<any>(`${this.datasetsUrl}/${id}`).pipe(
      map((dataset) => dataset?.name || 'Unknown Dataset'),
      catchError(() => of('Unknown Dataset'))
    );
  }

  /**
   * Get agent name by ID
   */
  private getAgentName(id: string): Observable<string> {
    return this.httpClient.get<any>(`${this.agentsUrl}/${id}`).pipe(
      map((agent) => agent?.name || 'Unknown Agent'),
      catchError(() => of('Unknown Agent'))
    );
  }

  /**
   * Get prompt name by ID
   */
  private getPromptName(id: string): Observable<string> {
    return this.httpClient.get<any>(`${this.promptsUrl}/${id}`).pipe(
      map((prompt) => prompt?.name || 'Default Prompt'),
      catchError(() => of('Default Prompt'))
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
   * Get comparison metrics - directly maps to API response structure
   */
  getComparisonMetrics(id: string): Observable<MetricDifference[]> {
    return this.httpClient
      .get<Comparison>(`${this.baseUrl}/${id}`) // Get the full comparison
      .pipe(
        map((comparison) => {
          // Extract and transform metric differences from the comparison results
          if (comparison.comparison_results?.metric_comparison) {
            return this.extractMetricDifferences(
              comparison.comparison_results.metric_comparison
            );
          }
          return [];
        }),
        catchError((error) => {
          return this.handleError(
            `Failed to fetch metrics for comparison with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Extract metric differences from the API response format
   */
  private extractMetricDifferences(metricComparison: any): MetricDifference[] {
    const metrics: MetricDifference[] = [];

    // Loop through each metric in the comparison
    for (const [metricName, data] of Object.entries(metricComparison)) {
      const metricData = data as any;

      metrics.push({
        name: metricName,
        metric_name: metricName,
        evaluation_a_value: metricData.evaluation_a?.average || 0,
        evaluation_b_value: metricData.evaluation_b?.average || 0,
        absolute_difference: metricData.comparison?.absolute_difference || 0,
        percentage_change: metricData.comparison?.percentage_change || 0,
        is_improvement: metricData.comparison?.is_improvement || false,
      });
    }

    return metrics;
  }

  /**
   * Get visualization data with fallback support
   */
  getVisualizationDataWithFallback(
    id: string,
    type: 'radar' | 'bar' | 'line'
  ): Observable<ApiVisualizationData | VisualizationData | null> {
    // First try to get the real data from the API
    return this.httpClient
      .get<ApiVisualizationData>(`${this.baseUrl}/${id}/visualizations/${type}`)
      .pipe(
        catchError((error) => {
          console.warn(
            `Visualization API endpoint not available for ${type}:`,
            error
          );
          // If API endpoint not available, try to build visualization from metrics
          return this.buildVisualizationFromMetrics(id, type);
        })
      );
  }

  /**
   * Get visualization data - use actual API or fallback to mock data
   */
  getVisualizationData(
    id: string,
    type: 'radar' | 'bar' | 'line'
  ): Observable<VisualizationData> {
    // First try to get the real data from the API
    return this.httpClient
      .get<any>(`${this.baseUrl}/${id}/visualizations/${type}`)
      .pipe(
        catchError((error) => {
          // If API endpoint not available, try to build visualization from metrics
          if (error.status === 404) {
            return this.buildVisualizationFromMetrics(id, type);
          }
          return this.handleError(
            `Failed to fetch ${type} visualization data for comparison with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Build visualization data from metrics if the API doesn't provide it
   */
  private buildVisualizationFromMetrics(
    id: string,
    type: 'radar' | 'bar' | 'line'
  ): Observable<VisualizationData> {
    // Get the comparison data which includes metrics
    return this.getComparison(id).pipe(
      map((comparison) => {
        // Extract metric differences
        let metrics: MetricDifference[] = [];

        if (comparison.comparison_results?.['metric_comparison']) {
          metrics = this.extractMetricDifferences(
            comparison.comparison_results['metric_comparison']
          );
        }

        // Now create visualization data
        return this.createVisualizationData(metrics, type);
      }),
      catchError((error) => {
        // Return mock data as a last resort if we can't get metrics
        console.log('Falling back to mock visualization data');
        return of(this.getMockVisualizationData(type));
      })
    );
  }

  /**
   * Create visualization data from metrics
   */
  private createVisualizationData(
    metrics: MetricDifference[],
    type: 'radar' | 'bar' | 'line'
  ): VisualizationData {
    const labels = metrics.map((m) => m.name || m.metric_name || '');

    // Create datasets based on visualization type
    let datasets;

    if (type === 'radar' || type === 'line') {
      datasets = [
        {
          label: 'Evaluation A',
          data: metrics.map((m) => m.evaluation_a_value),
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          borderColor: 'rgba(54, 162, 235, 1)',
          fill: type === 'radar',
        },
        {
          label: 'Evaluation B',
          data: metrics.map((m) => m.evaluation_b_value),
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          borderColor: 'rgba(255, 99, 132, 1)',
          fill: type === 'radar',
        },
      ];
    } else {
      // bar chart
      datasets = [
        {
          label: 'Evaluation A',
          data: metrics.map((m) => m.evaluation_a_value),
          backgroundColor: 'rgba(54, 162, 235, 0.7)',
        },
        {
          label: 'Evaluation B',
          data: metrics.map((m) => m.evaluation_b_value),
          backgroundColor: 'rgba(255, 99, 132, 0.7)',
        },
        {
          label: 'Difference (%)',
          data: metrics.map((m) => m.percentage_change || 0),
          backgroundColor: 'rgba(75, 192, 192, 0.7)',
        },
      ];
    }

    return {
      type,
      labels,
      datasets,
    };
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
  ): VisualizationData {
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
    return mockData;
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
