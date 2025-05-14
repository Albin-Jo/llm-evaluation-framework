import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';
import { HttpParams } from '@angular/common/http';

import { HttpClientService } from './common/http-client.service';

export interface DocumentContentResponse {
  dataset_id?: string;
  name?: string;
  type?: string;
  format?: string;
  content_type?: string;
  total_rows?: number;
  preview_rows?: number;
  content?: string | null;
  headers?: string[];
  rows?: string[];
  error?: string;
  key?: string;
}

@Injectable({
  providedIn: 'root'
})
export class DatasetContentService {
  private baseUrl = '__fastapi__/datasets';

  constructor(private httpClient: HttpClientService) {}

  /**
   * Retrieve dataset content preview
   * @param datasetId The ID of the dataset
   * @param limitRows Optional number of rows to limit in the preview
   * @returns Observable with dataset content preview
   */
  getDatasetContentPreview(datasetId: string, limitRows: number = 50): Observable<DocumentContentResponse | any[]> {
    const url = `${this.baseUrl}/${datasetId}/content`;
    
    // Set up appropriate parameters for content retrieval
    const params = new HttpParams()
      .set('limit_rows', limitRows.toString());
    
    console.log(`Requesting dataset content preview: ${url}`);
    
    return this.httpClient.get<any>(url, params)
      .pipe(
        tap(response => console.log('Received dataset content preview response type:', Array.isArray(response) ? 'array' : typeof response)),
        catchError(error => {
          console.error('Error retrieving dataset content:', error);
          // Return a default structure for error handling
          return of({ error: error.message || 'Failed to retrieve dataset content' });
        })
      );
  }

  /**
   * Retrieve document content from a dataset
   * @param datasetId The ID of the dataset containing the document
   * @param documentId The ID of the document to retrieve
   * @returns Observable with document content
   */
  getDocumentContent(datasetId: string, documentId: string): Observable<DocumentContentResponse | any[]> {
    const url = `${this.baseUrl}/${datasetId}/content`;
    
    console.log(`Requesting document content: ${url}`);
    
    return this.httpClient.get<any>(url)
      .pipe(
        tap(response => {
          console.log('Received document content response type:', Array.isArray(response) ? 'array' : typeof response);
          if (Array.isArray(response)) {
            console.log('Array response length:', response.length);
            if (response.length > 0) {
              console.log('First item type:', typeof response[0]);
              if (typeof response[0] === 'object') {
                console.log('Sample properties:', Object.keys(response[0]).slice(0, 5));
              }
            }
          }
        }),
        catchError(error => {
          console.error('Error retrieving document content:', error);
          // Return a default structure for error handling
          return of({ error: error.message || 'Failed to retrieve document content' });
        })
      );
  }

  /**
   * Parse CSV content into structured data
   * @param csvContent Raw CSV content as string
   * @returns Object with headers and data rows
   */
  parseCSVContent(csvContent: string): { headers: string[], data: any[] } {
    if (!csvContent || typeof csvContent !== 'string') {
      return { headers: [], data: [] };
    }

    try {
      // Split content into rows
      const rows = csvContent.split(/\r?\n/);
      if (rows.length === 0) {
        return { headers: [], data: [] };
      }

      // Extract headers from first row
      const headers = rows[0].split(',').map(header => header.trim());
      
      // Parse data rows
      const data = [];
      for (let i = 1; i < rows.length; i++) {
        const row = rows[i].trim();
        if (row) {
          // This is a simple parsing logic and might need enhancement
          // for handling quoted values, escaped commas, etc.
          const values = row.split(',');
          const rowData: Record<string, string> = {};
          
          headers.forEach((header, index) => {
            rowData[header] = values[index]?.trim() || '';
          });
          
          data.push(rowData);
        }
      }

      return { headers, data };
    } catch (e) {
      console.error('Error parsing CSV content:', e);
      return { headers: [], data: [] };
    }
  }

  /**
   * Create a structured response from an array of objects
   * @param arrayData Array of objects
   * @returns Structured document content response
   */
  createResponseFromArray(arrayData: any[]): DocumentContentResponse {
    if (!arrayData || !Array.isArray(arrayData) || arrayData.length === 0) {
      return {
        content: JSON.stringify(arrayData, null, 2),
        headers: [],
        rows: []
      };
    }

    // Extract headers from the first item that is an object
    const firstObjectIndex = arrayData.findIndex(item => item && typeof item === 'object' && !Array.isArray(item));
    
    if (firstObjectIndex === -1) {
      // No objects found, just return the raw content
      return {
        content: JSON.stringify(arrayData, null, 2),
        headers: [],
        rows: []
      };
    }

    // Get all unique keys from all objects in the array
    const headers = new Set<string>();
    arrayData.forEach(item => {
      if (item && typeof item === 'object' && !Array.isArray(item)) {
        Object.keys(item).forEach(key => headers.add(key));
      }
    });

    return {
      content: JSON.stringify(arrayData, null, 2),
      headers: Array.from(headers),
      rows: arrayData,
      total_rows: arrayData.length,
      preview_rows: arrayData.length
    };
  }
}