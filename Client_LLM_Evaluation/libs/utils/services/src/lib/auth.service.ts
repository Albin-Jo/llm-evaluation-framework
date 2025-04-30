import { inject, Injectable } from '@angular/core';
import { OidcSecurityService } from 'angular-auth-oidc-client';

@Injectable({
  providedIn: 'root',
})
export class AuthenticationService {
  private readonly oidcSecurityService = inject(OidcSecurityService);

  logout() {
    this.oidcSecurityService.logoff().subscribe((result) => console.log(result, 'logoff'));
    this.oidcSecurityService.logoffLocal();
  }
}
