import { AutoLoginPartialRoutesGuard } from 'angular-auth-oidc-client';
import { AuthGuard, RoleGuard } from '@ngtx-apps/utils/services';
import { Routes } from '@angular/router';
import { PATHS } from '@ngtx-apps/utils/shared';

export const APP_ROUTES: Routes = [
  {
    path: PATHS.EMPTY,
    pathMatch: 'full',
    redirectTo: PATHS.APP,
  },
  {
    path: PATHS.APP,
    title: 'Home',
    loadChildren: () => import('@ngtx-apps/feature/layout').then((m) => m.layoutRoutes),
    canActivate: [AutoLoginPartialRoutesGuard],
    canActivateChild: [RoleGuard, AuthGuard],
  },
  {
    path: PATHS.FORBIDDEN,
    title: 'Forbidden',
    loadComponent: () => import('@ngtx-apps/ui/components').then((m) => m.LoginForbiddenComponent),
  },
  {
    path: PATHS.UNATHORIZED,
    title: 'Unauthorized',
    loadComponent: () => import('@ngtx-apps/ui/components').then((m) => m.LoginUnauthorizedComponent),
  },
  {
    path: PATHS.LOGOUT,
    title: 'Logout',
    loadComponent: () => import('@ngtx-apps/ui/components').then((m) => m.LogoutComponent),
  },
  {
    path: PATHS.CALLBACK,
    title: 'Initialization',
    loadComponent: () => import('@ngtx-apps/ui/components').then((m) => m.LoginValidatorComponent), // does nothing but setting up auth
  },
  {
    path: PATHS.NOT_FOUND,
    title: 'Not Found',
    loadComponent: () => import('@ngtx-apps/ui/components').then((m) => m.NotfoundComponent),
  },
];
