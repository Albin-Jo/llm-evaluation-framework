/* Path: libs/data-access/services/src/lib/common/interceptors/auth.interceptor.ts */
import {
  HttpEvent,
  HttpHandler,
  HttpHeaders,
  HttpInterceptor,
  HttpRequest,
} from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import { StorageService } from '@ngtx-apps/utils/services';
import { AppConstant } from '@ngtx-apps/utils/shared';

@Injectable({
  providedIn: 'root',
})
export class AuthInterceptor implements HttpInterceptor {
  private readonly storageService = inject(StorageService);

  intercept(
    req: HttpRequest<any>,
    next: HttpHandler
  ): Observable<HttpEvent<any>> {
    console.log('AuthInterceptor processing request:', req.url);

    if (req.url.includes(AppConstant.KEYCLOCK_AUTHORITY)) {
      console.log('Skipping token for Keycloak request');
      return next.handle(req);
    } else {
      return this.handleTokenAuthentication(req, next);
    }
  }

  private handleTokenAuthentication(
    req: HttpRequest<any>,
    next: HttpHandler
  ): Observable<HttpEvent<any>> {
    let token = this.storageService.get('token');
    token = token ? 'Bearer ' + token : null;
    console.log('Token present:', !!token);

    // Check if this is a FormData request (file upload)
    const isFormData = req.body instanceof FormData;
    console.log('Is FormData request:', isFormData);

    let updatedHeaders = req.headers;

    if (!isFormData) {
      // Only set content-type for non-FormData requests
      console.log('Setting content-type for JSON request');
      updatedHeaders = updatedHeaders.set('content-type', 'application/json');
    } else {
      // For FormData, we need to ensure content-type is NOT set
      // as the browser will set it automatically with the correct boundary
      console.log('Removing content-type for FormData request');
      updatedHeaders = updatedHeaders.delete('content-type');
    }

    // Add CORS header
    updatedHeaders = updatedHeaders.set('Access-Control-Allow-Origin', '*');

    // Add token if available
    if (token) {
      updatedHeaders = updatedHeaders.set('Authorization', token);
    }

    const isAbsoluteUrl = req.url.startsWith('http://') || req.url.startsWith('https://');
    const isFastApiRequest = req.url.startsWith('__fastapi__/');
    let url: string;

    if (isAbsoluteUrl) {
      // Don't modify absolute URLs
      url = req.url;
      console.log('Using absolute URL:', url);
    } else if (isFastApiRequest) {
      // Remove the special prefix and prepend the FastAPI URL
      url = AppConstant.API_URL + req.url.substring('__fastapi__'.length);
      console.log('FastAPI request transformed:', req.url, '->', url);
    } else {
      // Regular API request - use the standard BASE_API_URL
      url = AppConstant.BASE_API_URL + req.url;
      console.log('Regular API request transformed:', req.url, '->', url);
    }

    const changedReq = req.clone({
      url: url,
      headers: updatedHeaders,
    });

    console.log('Final request URL:', changedReq.url);
    console.log('Final request method:', changedReq.method);

    if (isFormData) {
      console.log('FormData request - headers:', changedReq.headers.keys());
    }

    return next.handle(changedReq).pipe(
      tap(
        event => console.log('Request success type:', event.type),
        error => console.error('Request error:', error)
      )
    );
  }
}
