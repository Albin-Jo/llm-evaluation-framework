import { ActivatedRouteSnapshot, CanActivateChild, Router, RouterStateSnapshot, UrlTree } from '@angular/router';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { RoleService } from '../role.service';
import { ROLES } from '@ngtx-apps/utils/shared';

@Injectable({
  providedIn: 'root'
})
export class RoleGuard implements CanActivateChild {
  private readonly roleService = inject(RoleService)
  private readonly router = inject(Router)

  canActivateChild(childRoute: ActivatedRouteSnapshot, state: RouterStateSnapshot): boolean | UrlTree | Observable<boolean | UrlTree> | Promise<boolean | UrlTree> {
    return this.activateRoute(childRoute, state);
  }

  /**Check whether the route can be activated*/
  async activateRoute(_route: any, state: RouterStateSnapshot): Promise<any> {
    const routeObj = RouteToRoleMap.filter((route) => {
      if (state.url == route.path || state.url.includes(route.path) || route.path.includes(state.url)) {
        return true
      }
      return false
    })[0];

    let canActivate = false;

    if (routeObj) {
      for (const role of routeObj.roles) {
        if (role == ROLES.ALL) {
          canActivate = true;

        } else {
          const hasRole = await this.roleService.hasRole(role)
          if (hasRole) {
            canActivate = true;
          }
        }
      }
    }
    if (!canActivate) {
      this.router.navigate(['/notfound']);
    }
    return canActivate
  }
}

const RouteToRoleMap = [
  {
    path: '/login',
    roles: [ROLES.ALL]
  },
  {
    path: '/app/',
    roles: [ROLES.ALL]
  }
]
