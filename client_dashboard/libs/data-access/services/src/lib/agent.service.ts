/* Path: libs/data-access/services/src/lib/agent.service.ts */
import { Injectable } from '@angular/core';
import { Observable, throwError } from 'rxjs';
import { catchError, map, tap } from 'rxjs/operators';
import { HttpParams, HttpErrorResponse } from '@angular/common/http';
import {
  Agent,
  AgentFilterParams,
  AgentListResponse,
  AgentCreate,
  AgentUpdate,
  AgentResponse
} from '@ngtx-apps/data-access/models';
import { environment } from '@ngtx-apps/utils/shared';
import { HttpClientService } from './common/http-client.service';

@Injectable({
  providedIn: 'root'
})
export class AgentService {
  // Use the API endpoint path that matches FastAPI routes
  private baseUrl = '__fastapi__/agents';

  constructor(private httpClient: HttpClientService) {
    console.log('AgentService initialized with baseUrl:', this.baseUrl);
  }

  /**
   * Get a list of agents with optional filtering
   */
  getAgents(filters: AgentFilterParams = {}): Observable<AgentListResponse> {
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

    if (filters.domain) {
      params = params.set('domain', filters.domain);
    }

    if (filters.is_active !== undefined) {
      params = params.set('is_active', filters.is_active.toString());
    }

    if (filters.name) {
      params = params.set('name', filters.name);
    }

    console.log('Fetching agents with params:', params.toString());

    return this.httpClient.get<any>(this.baseUrl, params)
      .pipe(
        tap(response => console.log('Raw agent list response:', response)),
        map(response => {
          // Transform the API response to match the expected structure
          if (Array.isArray(response)) {
            console.log('Response is an array, transforming');
            return {
              agents: response,
              totalCount: response.length
            } as AgentListResponse;
          } else if (response.items && Array.isArray(response.items)) {
            // Handle paginated response format
            return {
              agents: response.items,
              totalCount: response.total || response.items.length
            } as AgentListResponse;
          }

          // Return the response as is if it already matches our format
          console.log('Returning response as is');
          return response as AgentListResponse;
        }),
        catchError(error => this.handleError('Failed to fetch agents', error))
      );
  }

  /**
   * Get a single agent by ID
   */
  getAgent(id: string): Observable<AgentResponse> {
    console.log(`Fetching agent with ID: ${id}`);
    console.log(`Full URL being called: ${this.baseUrl}/${id}`);

    return this.httpClient.get<AgentResponse>(`${this.baseUrl}/${id}`)
      .pipe(
        tap(response => console.log('Raw agent detail response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to fetch agent with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to fetch agent with ID ${id}`, error);
        })
      );
  }

  /**
   * Create a new agent
   */
  createAgent(agent: AgentCreate): Observable<AgentResponse> {
    console.log('Creating agent:', agent);

    return this.httpClient.post<AgentResponse>(this.baseUrl, agent)
      .pipe(
        tap(response => console.log('Create agent response:', JSON.stringify(response))),
        catchError(error => {
          console.error('Failed to create agent', error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError('Failed to create agent', error);
        })
      );
  }

  /**
   * Update an existing agent
   */
  updateAgent(id: string, agent: AgentUpdate): Observable<AgentResponse> {
    console.log(`Updating agent ${id}:`, JSON.stringify(agent));

    return this.httpClient.put<AgentResponse>(`${this.baseUrl}/${id}`, agent)
      .pipe(
        tap(response => console.log('Update agent response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to update agent with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to update agent with ID ${id}`, error);
        })
      );
  }

  /**
   * Delete an agent
   */
  deleteAgent(id: string): Observable<void> {
    console.log(`Deleting agent with ID: ${id}`);
    console.log(`Delete URL: ${this.baseUrl}/${id}`);

    return this.httpClient.delete<void>(`${this.baseUrl}/${id}`)
      .pipe(
        tap(() => console.log(`Agent ${id} deleted successfully`)),
        catchError(error => {
          console.error(`Failed to delete agent with ID ${id}`, error);
          console.error('Delete error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to delete agent with ID ${id}`, error);
        })
      );
  }

  /**
   * Test an agent with sample input
   */
  testAgent(id: string, testInput: Record<string, any>): Observable<Record<string, any>> {
    console.log(`Testing agent with ID: ${id}`);
    console.log(`Test input:`, JSON.stringify(testInput));

    return this.httpClient.post<Record<string, any>>(`${this.baseUrl}/${id}/test`, testInput)
      .pipe(
        tap(response => console.log('Test agent response:', JSON.stringify(response))),
        catchError(error => {
          console.error(`Failed to test agent with ID ${id}`, error);
          console.error('Error details:', JSON.stringify(error, null, 2));
          return this.handleError(`Failed to test agent with ID ${id}`, error);
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
