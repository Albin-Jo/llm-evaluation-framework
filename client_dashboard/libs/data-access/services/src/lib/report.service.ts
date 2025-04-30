/* Path: libs/data-access/services/src/lib/report.service.ts */
import { Injectable } from '@angular/core';
import { Observable, throwError } from 'rxjs';
import { HttpParams, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { catchError, map, tap } from 'rxjs/operators';
import {
  Report,
  ReportCreate,
  ReportUpdate,
  ReportFilterParams,
  ReportListResponse,
  ReportStatus
} from '@ngtx-apps/data-access/models';
import { AppConstant } from '@ngtx-apps/utils/shared';
import { HttpClientService } from './common/http-client.service';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class ReportService {
  private baseUrl = '__fastapi__/reports';

  constructor(
    private httpClient: HttpClientService,
    private http: HttpClient // For blob downloads
  ) {
    console.log('ReportService initialized with baseUrl:', this.baseUrl);
  }

  /**
   * Get a list of reports with optional filtering
   * @param filters Optional filter parameters
   * @returns Observable of ReportListResponse
   */
  getReports(filters: ReportFilterParams = {}): Observable<ReportListResponse> {
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

    if (filters.format) {
      params = params.set('format', filters.format);
    }

    if (filters.evaluation_id) {
      params = params.set('evaluation_id', filters.evaluation_id);
    }

    if (filters.name) {
      params = params.set('name', filters.name);
    }

    if (filters.is_public !== undefined) {
      params = params.set('is_public', filters.is_public.toString());
    }

    console.log('Fetching reports with params:', params.toString());

    return this.httpClient.get<any>(this.baseUrl, params)
      .pipe(
        tap(response => console.log('Raw report list response:', response)),
        map(response => {
          // Transform the API response to match the expected structure
          if (Array.isArray(response)) {
            console.log('Response is an array, transforming');
            return {
              reports: response,
              totalCount: response.length
            } as ReportListResponse;
          } else if (response.items && Array.isArray(response.items)) {
            // Handle paginated response format
            return {
              reports: response.items,
              totalCount: response.total || response.items.length
            } as ReportListResponse;
          }

          // Return the response as is if it already matches our format
          console.log('Returning response as is');
          return response as ReportListResponse;
        }),
        catchError(error => this.handleError('Failed to fetch reports', error))
      );
  }

  /**
   * Get a report by ID
   * @param id The ID of the report to retrieve
   * @returns Observable of Report
   */
  getReport(id: string): Observable<Report> {
    console.log(`Fetching report with ID: ${id}`);
    console.log(`Full URL being called: ${this.baseUrl}/${id}`);

    return this.httpClient.get<Report>(`${this.baseUrl}/${id}`)
      .pipe(
        tap(response => console.log('Report detail response:', response)),
        catchError(error => {
          console.error(`Failed to fetch report with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to fetch report with ID ${id}`, error);
        })
      );
  }

  /**
   * Create a new report
   * @param report The report data to create
   * @returns Observable of Report
   */
  createReport(report: ReportCreate): Observable<Report> {
    console.log('Creating report:', report);

    return this.httpClient.post<Report>(this.baseUrl, report)
      .pipe(
        tap(response => console.log('Report creation response:', response)),
        catchError(error => {
          console.error('Failed to create report', error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError('Failed to create report', error);
        })
      );
  }

  /**
   * Update an existing report
   * @param id The ID of the report to update
   * @param report The data to update
   * @returns Observable of Report
   */
  updateReport(id: string, report: ReportUpdate): Observable<Report> {
    console.log(`Updating report ${id}:`, report);

    return this.httpClient.put<Report>(`${this.baseUrl}/${id}`, report)
      .pipe(
        tap(response => console.log('Report update response:', response)),
        catchError(error => {
          console.error(`Failed to update report with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to update report with ID ${id}`, error);
        })
      );
  }

  /**
   * Delete a report
   * @param id The ID of the report to delete
   * @returns Observable of void
   */
  deleteReport(id: string): Observable<void> {
    console.log(`Deleting report with ID: ${id}`);
    console.log(`Delete URL: ${this.baseUrl}/${id}`);

    return this.httpClient.delete<void>(`${this.baseUrl}/${id}`)
      .pipe(
        tap(() => console.log(`Report ${id} deleted successfully`)),
        catchError(error => {
          console.error(`Failed to delete report with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to delete report with ID ${id}`, error);
        })
      );
  }

  /**
   * Generate a report (convert from draft to generated status)
   * @param id The ID of the report to generate
   * @returns Observable of Report
   */
  generateReport(id: string): Observable<Report> {
    console.log(`Generating report with ID: ${id}`);
    console.log(`Generate URL: ${this.baseUrl}/${id}/generate`);

    return this.httpClient.post<Report>(`${this.baseUrl}/${id}/generate`, {})
      .pipe(
        tap(response => console.log('Generate report response:', response)),
        catchError(error => {
          console.error(`Failed to generate report with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to generate report with ID ${id}`, error);
        })
      );
  }

  /**
   * Download a report
   * @param id The ID of the report to download
   * @returns Observable of Blob
   */
  downloadReport(id: string): Observable<Blob> {
    console.log(`Downloading report with ID: ${id}`);

    // Since HttpClientService doesn't have a getBlob method,
    // we'll use the direct HttpClient with the transformed URL

    // Construct the URL with the same pattern used in the interceptor
    const url = AppConstant.API_URL + `/reports/${id}/download`;
    console.log(`Download URL: ${url}`);

    // Handle authentication manually for this special case
    let headers = new HttpHeaders();
    const token = localStorage.getItem('token');
    if (token) {
      headers = headers.set('Authorization', `Bearer ${token}`);
    }

    return this.http.get(url, {
      headers: headers,
      responseType: 'blob'
    }).pipe(
      tap(() => console.log(`Report ${id} downloaded successfully`)),
      catchError(error => {
        console.error(`Failed to download report with ID ${id}`, error);
        console.error('Error details:', JSON.stringify(error, null, 2));
        return this.handleError(`Failed to download report with ID ${id}`, error);
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