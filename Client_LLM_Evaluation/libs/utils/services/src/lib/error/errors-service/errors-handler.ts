import { HttpErrorResponse } from '@angular/common/http';
import { ErrorHandler, inject, Injectable } from '@angular/core';
import { ErrorsService } from './errors-service';
@Injectable()
export class ErrorsHandler implements ErrorHandler {
  private readonly errorService = inject(ErrorsService);
  handleError(error: Error | HttpErrorResponse) {
    this.errorService.log(error).subscribe(() => {
      console.log('logged')
    });
  }
}
