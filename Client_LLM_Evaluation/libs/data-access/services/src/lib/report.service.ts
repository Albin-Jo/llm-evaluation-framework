import { Injectable } from '@angular/core';
import { Observable, throwError } from 'rxjs';
import {
  HttpParams,
  HttpErrorResponse,
  HttpHeaders,
} from '@angular/common/http';
import { catchError, map, tap } from 'rxjs/operators';
import {
  Report,
  ReportCreate,
  ReportUpdate,
  ReportFilterParams,
  ReportListResponse,
  ReportStatus,
} from '@ngtx-apps/data-access/models';
import { AppConstant } from '@ngtx-apps/utils/shared';
import { HttpClientService } from './common/http-client.service';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root',
})
export class ReportService {
  private baseUrl = '__fastapi__/reports';

  constructor(
    private httpClient: HttpClientService,
    private http: HttpClient // For blob downloads
  ) {}

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
      params = params.set(
        'skip',
        ((filters.page - 1) * (filters.limit || 10)).toString()
      );
    }

    if (filters.limit !== undefined) {
      params = params.set('limit', filters.limit.toString());
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

    return this.httpClient.get<any>(this.baseUrl, params).pipe(
      map((response) => {
        // Transform the API response to match the expected structure
        if (Array.isArray(response)) {
          return {
            reports: response,
            totalCount: response.length,
          } as ReportListResponse;
        }
        // If response has the expected structure
        else if (
          response &&
          response.reports &&
          response.totalCount !== undefined
        ) {
          return response as ReportListResponse;
        }
        // If response has items/total structure (like paginated response)
        else if (response && response.items && response.total !== undefined) {
          return {
            reports: response.items,
            totalCount: response.total,
          } as ReportListResponse;
        }
        // Fallback: return empty list
        return {
          reports: [],
          totalCount: 0,
        } as ReportListResponse;
      }),
      catchError((error) => this.handleError('Failed to fetch reports', error))
    );
  }

  /**
   * Get a report by ID
   * @param id The ID of the report to retrieve
   * @returns Observable of Report
   */
  getReport(id: string): Observable<Report> {
    return this.httpClient.get<Report>(`${this.baseUrl}/${id}`).pipe(
      catchError((error) => {
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
    return this.httpClient.post<Report>(this.baseUrl, report).pipe(
      catchError((error) => {
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
    return this.httpClient.put<Report>(`${this.baseUrl}/${id}`, report).pipe(
      catchError((error) => {
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
    return this.httpClient.delete<void>(`${this.baseUrl}/${id}`).pipe(
      catchError((error) => {
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
    return this.httpClient
      .post<Report>(`${this.baseUrl}/${id}/generate`, {})
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to generate report with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Download a report
   * @param id The ID of the report to download
   * @returns Observable of Blob
   */
  downloadReport(id: string): Observable<Blob> {
    // Construct the URL with the same pattern used in the interceptor
    const url = `${AppConstant.API_URL}/reports/${id}/download`;

    // Handle authentication manually for this special case
    let headers = new HttpHeaders();
    const token = localStorage.getItem('token');
    if (token) {
      headers = headers.set('Authorization', `Bearer ${token}`);
    }

    return this.http
      .get(url, {
        headers: headers,
        responseType: 'blob',
      })
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to download report with ID ${id}`,
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

  /**
   * Preview a report in HTML format
   * @param id The ID of the report to preview
   * @returns Observable of HTML string
   */
  previewReport(id: string): Observable<string> {
    // Use the direct HttpClient for this request since we want the raw HTML
    // Construct the URL with the same pattern used in the interceptor
    const url = `${AppConstant.API_URL}/reports/${id}/preview`;

    // Handle authentication manually for this special case
    let headers = new HttpHeaders();
    const token = localStorage.getItem('token');
    if (token) {
      headers = headers.set('Authorization', `Bearer ${token}`);
    }

    return this.http
      .get(url, {
        headers: headers,
        responseType: 'text',
      })
      .pipe(
        catchError((error) => {
          return this.handleError(
            `Failed to preview report with ID ${id}`,
            error
          );
        })
      );
  }

  /**
   * Send a report via email
   * @param id The ID of the report to send
   * @param recipients Array of email recipients
   * @param subject Optional email subject
   * @param message Optional email message
   * @returns Observable of success response
   */
  sendReport(
    id: string,
    recipients: { email: string; name?: string }[],
    subject?: string,
    message?: string
  ): Observable<any> {
    const requestData = {
      recipients: recipients,
      subject: subject,
      message: message,
      include_pdf: true,
    };

    return this.httpClient
      .post<any>(`${this.baseUrl}/${id}/send`, requestData)
      .pipe(
        catchError((error) => {
          return this.handleError(`Failed to send report with ID ${id}`, error);
        })
      );
  }
}
