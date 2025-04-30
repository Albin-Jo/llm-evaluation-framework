import { CommonModule } from '@angular/common';
import { NgModule, ErrorHandler } from '@angular/core';
import { RouterModule } from '@angular/router';
import { ErrorsComponent } from '@ngtx-apps/ui/components';
import { ErrorsHandler } from './errors-service/errors-handler';
import { ErrorRoutingModule } from './errors-routing.module';
import { ErrorsService } from './errors-service/errors-service';

@NgModule({
    imports: [CommonModule, RouterModule, ErrorRoutingModule, ErrorsComponent],
    providers: [
        ErrorsService,
        {
            provide: ErrorHandler,
            useClass: ErrorsHandler,
        },
    ],
})
export class ErrorsModule {}
