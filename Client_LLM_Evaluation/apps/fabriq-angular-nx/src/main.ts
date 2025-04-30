import { enableProdMode } from '@angular/core';
import { appConfig } from './app/app.config';
import { bootstrapApplication } from '@angular/platform-browser';
import { environment } from '@ngtx-apps/utils/shared';
import { AppComponent } from './app/app.component';
import { registerLicense } from '@syncfusion/ej2-base';
import { defineCustomElements as defineStencilElements } from '@qrwc/qrsc-enterprise/loader'

// Registering Syncfusion license key
registerLicense('ORg4AjUWIQA/Gnt2UFhhQlJBfV5AQmBIYVp/TGpJfl96cVxMZVVBJAtUQF1hTH5RdkxjX3xZcnZdRGBd');

if (environment.production) {
  enableProdMode();
}

bootstrapApplication(AppComponent, appConfig).catch((err) => console.log(err));
defineStencilElements(window);
