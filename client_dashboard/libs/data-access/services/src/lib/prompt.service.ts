/* Path: libs/data-access/services/src/lib/prompt.service.ts */
import { Injectable } from '@angular/core';
import { Observable, throwError } from 'rxjs';
import { HttpParams, HttpErrorResponse } from '@angular/common/http';
import { catchError, map, tap } from 'rxjs/operators';
import {
  PromptCreate,
  PromptFilter,
  PromptResponse,
  PromptUpdate
} from '@ngtx-apps/data-access/models';
import { environment } from '@ngtx-apps/utils/shared';
import { HttpClientService } from './common/http-client.service';

@Injectable({
  providedIn: 'root'
})
export class PromptService {
  private baseUrl = '__fastapi__/prompts';

  constructor(private httpClient: HttpClientService) {
    console.log('PromptService initialized with baseUrl:', this.baseUrl);
  }

  /**
   * Get a list of prompts with optional filtering
   * @param filter Optional filter parameters
   * @returns Observable of PromptResponse array
   */
  getPrompts(filter: PromptFilter = {}): Observable<PromptResponse[]> {
    let params = new HttpParams();

    // Set default pagination if not provided
    params = params.set('skip', filter.skip?.toString() || '0');
    params = params.set('limit', filter.limit?.toString() || '100');

    // Add other filter parameters if they exist
    if (filter.is_public !== undefined) {
      params = params.set('is_public', filter.is_public.toString());
    }

    if (filter.template_id) {
      params = params.set('template_id', filter.template_id);
    }

    console.log('Fetching prompts with params:', params.toString());

    return this.httpClient.get<PromptResponse[]>(this.baseUrl, params)
      .pipe(
        tap(response => console.log('Prompts list response:', response)),
        catchError(error => this.handleError('Failed to fetch prompts', error))
      );
  }

  /**
   * Get a prompt by ID
   * @param promptId The ID of the prompt to retrieve
   * @returns Observable of PromptResponse
   */
  getPromptById(promptId: string): Observable<PromptResponse> {
    console.log(`Fetching prompt with ID: ${promptId}`);
    console.log(`Full URL being called: ${this.baseUrl}/${promptId}`);

    return this.httpClient.get<PromptResponse>(`${this.baseUrl}/${promptId}`)
      .pipe(
        tap(response => console.log('Prompt detail response:', response)),
        catchError(error => {
          console.error(`Failed to fetch prompt with ID ${promptId}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to fetch prompt with ID ${promptId}`, error);
        })
      );
  }

  /**
   * Create a new prompt
   * @param prompt The prompt data to create
   * @returns Observable of PromptResponse
   */
  createPrompt(prompt: PromptCreate): Observable<PromptResponse> {
    console.log('Creating prompt:', prompt);

    return this.httpClient.post<PromptResponse>(this.baseUrl, prompt)
      .pipe(
        tap(response => console.log('Prompt creation response:', response)),
        catchError(error => {
          console.error('Failed to create prompt', error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError('Failed to create prompt', error);
        })
      );
  }

  /**
   * Update an existing prompt
   * @param promptId The ID of the prompt to update
   * @param updateData The data to update
   * @returns Observable of PromptResponse
   */
  updatePrompt(promptId: string, updateData: PromptUpdate): Observable<PromptResponse> {
    console.log(`Updating prompt ${promptId}:`, updateData);

    return this.httpClient.put<PromptResponse>(`${this.baseUrl}/${promptId}`, updateData)
      .pipe(
        tap(response => console.log('Prompt update response:', response)),
        catchError(error => {
          console.error(`Failed to update prompt with ID ${promptId}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to update prompt with ID ${promptId}`, error);
        })
      );
  }

  /**
   * Delete a prompt
   * @param promptId The ID of the prompt to delete
   * @returns Observable of void
   */
  deletePrompt(promptId: string): Observable<void> {
    console.log(`Deleting prompt with ID: ${promptId}`);
    console.log(`Delete URL: ${this.baseUrl}/${promptId}`);

    return this.httpClient.delete<void>(`${this.baseUrl}/${promptId}`)
      .pipe(
        tap(() => console.log(`Prompt ${promptId} deleted successfully`)),
        catchError(error => {
          console.error(`Failed to delete prompt with ID ${promptId}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to delete prompt with ID ${promptId}`, error);
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
