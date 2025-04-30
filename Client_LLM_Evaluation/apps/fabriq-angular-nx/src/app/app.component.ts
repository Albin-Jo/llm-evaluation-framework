import { Component, inject, OnInit } from '@angular/core';
import { OidcSecurityService } from 'angular-auth-oidc-client';
import { Router, RouterModule } from '@angular/router';
import { StorageService, AnalyticService } from '@ngtx-apps/utils/services';
import { AuthenticationConstant, environment, PATHS } from '@ngtx-apps/utils/shared';

@Component({
  selector: 'fabriq-apps-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  imports: [RouterModule]
})
export class AppComponent implements OnInit {
  title = 'Welcome to Employee Experience';
  private readonly oidcSecurityService = inject(OidcSecurityService);
  private readonly storageservice = inject(StorageService);
  private readonly analyticService = inject(AnalyticService)
  private readonly router = inject(Router)
  constructor() {
    this.storageservice.set(AuthenticationConstant.runtimeMode, AuthenticationConstant.runtimeMode_WEB);
  }

  ngOnInit(): void {
    this.oidcSecurityService.checkAuth().subscribe({
      next: ({ isAuthenticated, userData, accessToken }) => {
        if (isAuthenticated) {
          this.storageservice.set('token', accessToken);
          this.storageservice.set('user', JSON.stringify(userData));
          if (environment.trackAnalytics){
            this.analyticService.init();
          }
        }
      },
      error: () => {
        this.router.navigate([PATHS.APP]);
      },
    });
  }
}
