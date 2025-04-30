import { inject, Injectable } from '@angular/core';
import { TrackingDetails } from '@ngtx-apps/data-access/models';
import { Observable } from 'rxjs';
import { HttpClientService } from './common/http-client.service';

@Injectable({
  providedIn: 'root',
})
export class TrackingService {
  private readonly httpClientService = inject(HttpClientService);

  /**
   * Method for calling the API for saving the analytical details
   */
  track(trackingDetails: TrackingDetails | any): Observable<any> {
    return this.httpClientService.post<any>('track', trackingDetails);
  }
}
