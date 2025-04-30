import { BodyData } from './types';
import { Observable } from 'rxjs';
import { inject, Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';

@Injectable({
  providedIn: 'root',
})
export class HttpClientService {
  /**
   * Create a new HttpClient instance with default configuration to call Restful Service
   */
  private readonly http = inject(HttpClient);

  /**
   * Performs `GET` request
   * @param path is the server resource path that will be used for the request
   * @param queryParams are the URL parameters / query string to be sent with the request. Must be a plain object or a `URLSearchParams` object
   * @param headers are custom headers to be sent
   * @returns An `Observable` of Http Client GET request instance
   */

  get<T = any>(
    path: string,
    queryParams?: HttpParams,
    headers?: HttpHeaders,
    cancel?: boolean
  ): Observable<any> {
    return this.http.get<T>(path, {
      headers: headers,
      params: queryParams,
    });
  }

  /**
   * Performs `POST` request
   * @param path is the server resource path that will be used for the request
   * @param body is the request method to be used when making the request
   * @param queryParams are the URL parameters / query string to be sent with the request. Must be a `plain object` or a `URLSearchParams` object
   * @param headers are custom headers to be sent
   * @returns An `Observable` of Http Client POST request instance
   */
  post<T = any>(
    path: string,
    body: BodyData,
    queryParams?: HttpParams,
    headers?: HttpHeaders
  ): Observable<T> {
    return this.http.post<T>(path, body, {
      headers: headers,
      params: queryParams,
    });
  }

  /**
   * Performs `PUT` request
   * @param path is the server resource path that will be used for the request
   * @param body is the request method to be used when making the request
   * @param queryParams are the URL parameters / query string to be sent with the request. Must be a `plain object` or a `URLSearchParams` object
   * @param headers are custom headers to be sent
   * @returns An `Observable` of Http Client PUT request instance
   */
  put<T = any>(
    path: string,
    body: BodyData,
    queryParams?: HttpParams,
    headers?: HttpHeaders
  ): Observable<T> {
    return this.http.put<T>(path, body, {
      headers: headers,
      params: queryParams,
    });
  }

  /**
   * Performs `DELETE` request
   * @param path is the server resource path that will be used for the request
   * @param queryParams are the URL parameters / query string to be sent with the request. Must be a `plain object` or a `URLSearchParams` object
   * @param headers are custom headers to be sent
   * @returns An `Observable` of HTTP Client DELETE request instance
   */
  delete<T = any>(
    path: string,
    queryParams?: HttpParams,
    headers?: HttpHeaders
  ): Observable<T> {
    return this.http.delete<T>(path, {
      headers: headers,
      params: queryParams,
    });
  }
}
