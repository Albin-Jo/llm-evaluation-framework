import { ApplicationConfig, ErrorHandler, isDevMode } from '@angular/core';
import { provideRouter, TitleStrategy, withComponentInputBinding, withViewTransitions } from '@angular/router';
import { HTTP_INTERCEPTORS, provideHttpClient, withInterceptors, withInterceptorsFromDi } from '@angular/common/http';
import { LogLevel, provideAuth, } from 'angular-auth-oidc-client';
import { AppConstant } from '@ngtx-apps/utils/shared';
import { ErrorsService, ErrorsHandler, CustomTitleStrategy } from '@ngtx-apps/utils/services';
import { provideServiceWorker } from '@angular/service-worker';
import { AuthInterceptor } from '@ngtx-apps/data-access/services';
import { provideNgIdle } from '@ng-idle/core';
import { Title } from '@angular/platform-browser';
import { APP_ROUTES } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(
      APP_ROUTES,
      withViewTransitions(),
      withComponentInputBinding(),
    ),

    provideHttpClient(withInterceptors([]), withInterceptorsFromDi()),
    {
      provide: HTTP_INTERCEPTORS,
      useClass: AuthInterceptor,
      multi: true,
    },
    provideAuth({
      config: {
        triggerAuthorizationResultEvent: true,
        postLoginRoute: 'app',
        forbiddenRoute: 'forbidden',
        unauthorizedRoute: 'unauthorized',
        logLevel: LogLevel.Error,
        historyCleanupOff: false,
        authority: AppConstant.KEYCLOCK_AUTHORITY,
        redirectUrl: AppConstant.KEYCLOCK_REDIRECTURL,
        postLogoutRedirectUri: AppConstant.KEYCLOCK_POSTLOGOUTREDIRECTURI,
        clientId: AppConstant.KEYCLOCK_CLIENTID,
        scope: AppConstant.KEYCLOCK_SCOPE,
        responseType: 'code',
        silentRenew: true,
        useRefreshToken: true,
        renewTimeBeforeTokenExpiresInSeconds: 5 * 60, //Renew before 5 min of expiry
        tokenRefreshInSeconds: 60, //Check expiry every one minute
        ignoreNonceAfterRefresh: true,
      },
    }),
    provideServiceWorker('ngsw-worker.js', {
      enabled: !isDevMode(),
      // Register the ServiceWorker as soon as the application is stable
      // or after 30 seconds (whichever comes first).
      registrationStrategy: 'registerWhenStable:30000',
    }),
    ErrorsService,
    {
      provide: ErrorHandler,
      useClass: ErrorsHandler,
    },
    Title,
    { provide: TitleStrategy, useClass: CustomTitleStrategy },
    provideNgIdle()
  ],
};
