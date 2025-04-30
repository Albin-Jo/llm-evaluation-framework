import { errorMessage } from '@ngtx-apps/data-access/models';
import { inject, Injectable } from '@angular/core';
import { HttpClientService } from './common/http-client.service';
@Injectable({
  providedIn: 'root',
})
export class LogService {
  private readonly http = inject(HttpClientService);
  LogError(error: any) {
    return this.http.post<errorMessage>('lgms/log/cerrlog', error);
  }
}
