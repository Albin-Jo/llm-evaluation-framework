import { Injectable } from '@angular/core';
import { Observable, throwError } from 'rxjs';
import { catchError, map, tap } from 'rxjs/operators';
import { HttpParams, HttpErrorResponse } from '@angular/common/http';
import {
  PromptResponse,
  PromptFilterParams,
  PromptCreate,
  PromptUpdate,
  PromptsResponse
} from '@ngtx-apps/data-access/models';
import { HttpClientService } from './common/http-client.service';

@Injectable({
  providedIn: 'root'
})
export class PromptService {
  // API endpoint path
  private baseUrl = '__fastapi__/prompts';

  constructor(private httpClient: HttpClientService) {
  }

  /**
   * Get a list of prompts with optional filtering
   */
  getPrompts(filters: PromptFilterParams = {}): Observable<PromptsResponse> {
    // Convert the filters object to HttpParams
    let params = new HttpParams();

    // Add each parameter if it exists
    // Convert page to skip parameter (for pagination)
    if (filters.page !== undefined) {
      params = params.set('skip', ((filters.page - 1) * (filters.limit || 10)).toString());
    } else {
      params = params.set('skip', '0');
    }

    // Add limit parameter for pagination
    if (filters.limit !== undefined) {
      params = params.set('limit', filters.limit.toString());
    } else {
      params = params.set('limit', '10'); // Default limit
    }

    // Add name filter (search term)
    if (filters.name) {
      params = params.set('name', filters.name);
    }

    // Add category filter
    if (filters.category) {
      params = params.set('category', filters.category);
    }

    // Add isPublic filter (convert boolean to string)
    if (filters.isPublic !== undefined) {
      params = params.set('is_public', filters.isPublic.toString());
    }

    // Add sort parameters - always sending these parameters for consistency
    params = params.set('sort_by', filters.sortBy || 'created_at');
    params = params.set('sort_dir', filters.sortDirection || 'desc');

    return this.httpClient.get<any>(this.baseUrl, params)
      .pipe(
        map(response => {
          // Transform the API response to match the expected structure
          if (response && response.items && Array.isArray(response.items)) {
            // The API returns {items: [...], total: number}
            return {
              prompts: response.items,
              totalCount: response.total
            } as PromptsResponse;
          } else if (Array.isArray(response)) {
            // Fallback for backward compatibility or different API response format
            return {
              prompts: response,
              totalCount: response.length
            } as PromptsResponse;
          }

          // Return the response as is if it already matches our format
          return response as PromptsResponse;
        }),
        catchError(error => this.handleError('Failed to fetch prompts', error))
      );
  }

  /**
   * Get a prompt by ID
   */
  getPrompt(id: string): Observable<PromptResponse> {
    return this.httpClient.get<PromptResponse>(`${this.baseUrl}/${id}`)
      .pipe(
        catchError(error => this.handleError(`Failed to fetch prompt with ID ${id}`, error))
      );
  }
  

  /**
   * Create a new prompt
   */
  createPrompt(prompt: PromptCreate): Observable<PromptResponse> {
    return this.httpClient.post<PromptResponse>(this.baseUrl, prompt)
      .pipe(
        catchError(error => this.handleError('Failed to create prompt', error))
      );
  }

  /**
   * Update an existing prompt
   */
  updatePrompt(id: string, prompt: PromptUpdate): Observable<PromptResponse> {
    return this.httpClient.put<PromptResponse>(`${this.baseUrl}/${id}`, prompt)
      .pipe(
        catchError(error => this.handleError(`Failed to update prompt with ID ${id}`, error))
      );
  }

/**
   * Get a prompt by ID (alias for getPrompt to maintain compatibility)
   */
 getPromptById(id: string): Observable<PromptResponse> {
  return this.getPrompt(id);
}
  deletePrompt(id: string): Observable<void> {
    return this.httpClient.delete<void>(`${this.baseUrl}/${id}`)
      .pipe(
        catchError(error => this.handleError(`Failed to delete prompt with ID ${id}`, error))
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