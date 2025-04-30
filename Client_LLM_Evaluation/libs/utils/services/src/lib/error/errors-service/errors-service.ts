import { inject, Injectable, Injector } from '@angular/core';
import { LocationStrategy, PathLocationStrategy } from '@angular/common';
import { Router, Event, NavigationError } from '@angular/router';

// import * as StackTraceParser from 'stacktrace-parser';
import { LogService } from '@ngtx-apps/data-access/services';
import { ErrorWithContext } from './error-context.interface';
import { environment } from '@ngtx-apps/utils/shared';

@Injectable()
export class ErrorsService {

  private readonly injector = inject(Injector);
  private readonly router = inject(Router);
  private readonly logService = inject(LogService);

  constructor() {
    // Subscribe to the NavigationError
    this.router.events.subscribe((event: Event) => {
      if (event instanceof NavigationError) {
        // Redirect to the ErrorComponent
        this.log(event.error).subscribe((errorWithContext) => {
          this.router.navigate(['/error'], { queryParams: errorWithContext });
        });
      }
    });
  }

  log(error: any) {
    const errorToSend = this.addContextInfo(error);
    let clientErrorMessage = 'Unknown client error occured';
    if (errorToSend.message) {
      clientErrorMessage = JSON.stringify(errorToSend);
    }
    console.error(clientErrorMessage);
    return this.logService.LogError({ message: clientErrorMessage });
  }

  addContextInfo(error: any) {
    // You can include context details here (usually coming from other services: UserService...)
    const name = error.name || null;
    const appId = environment.appLogId;
    const time = new Date().getTime();
    const location = this.injector.get(LocationStrategy);
    const url = location instanceof PathLocationStrategy ? location.path() : '';
    const status = error.status || null;
    const message = error.message || error.toString();
    const code = error.code || null;
    const response = error.response || null;
    let stack: any;
    if (error.stack) {
      // stack = StackTraceParser.parse(error.stack);
    }
    const errorWithContext: ErrorWithContext = {
      name,
      appId,
      time,
      url,
      status,
      message,
      stack,
      code,
      response,
    };
    return errorWithContext;
  }
}
