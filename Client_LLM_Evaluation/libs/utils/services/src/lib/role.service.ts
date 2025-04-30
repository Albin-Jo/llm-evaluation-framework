
import { inject, Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { StorageService } from './storage.service';
import { AuthenticationConstant } from '@ngtx-apps/utils/shared';

@Injectable({
  providedIn: 'root',
})
export class RoleService {

  private readonly storageService = inject(StorageService);

  private readonly roles: BehaviorSubject<Array<string>> = new BehaviorSubject<Array<string>>([]);
  public roles$ = this.roles.asObservable();
  /**
   *
   * @returns roles from jwt token
   */
  private async getTokenRoles(): Promise<string[]> {
    let tokenRoles: string[] = []
    this.roles$.subscribe((roles: string[]) => {
      tokenRoles = roles;
    });
    if (tokenRoles.length == 0) {
      try {
        const token: any = await this.storageService.get(AuthenticationConstant.token);
        const jwtData = token.split('.')[1]
        const decodedJwtJsonData = window.atob(jwtData)
        const decodedJwtData = JSON.parse(decodedJwtJsonData);

        // Initialize roles with Primary Role
        tokenRoles = [...decodedJwtData.realm_access.roles];

        // Add Secondary Role if it exists and is not empty
        const secondaryRoles = undefined;
        // if (secondaryRoles !== undefined) {
        //   tokenRoles.push(...secondaryRoles);
        // }

      } catch {
        tokenRoles = []
      }
      this.roles.next(tokenRoles);
    }
    return tokenRoles
  }

    /**
   *
   * @returns return a list of mixed roles
   */
    async getRoles() {
      const roles: string[] = await this.getTokenRoles();
      return roles
    }


  /**
   *
   * @param role
   * @returns Checks if the user token has the role
   */
  async hasRole(role: string) {
    const roles: string[] = await this.getTokenRoles();
    if (roles.includes(role)) {
      return true;
    }
    return false;
  }

  /**
  *
  * @param roles
  * @returns Checks if the user token has any one of the role
  */
  async hasAnyRole(rolesList: string[]) {
    const roles: string[] = await this.getTokenRoles();
    for (const role of rolesList) {
      if (roles.includes(role)) {
        return true;
      }
    }
    return false;
  }

}
