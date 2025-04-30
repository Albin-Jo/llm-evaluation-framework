import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { RoleService } from '../role.service';
import { StorageService } from '../storage.service';

export const AuthGuard: CanActivateFn = async (route, state) => {
  const storageSrv = inject(StorageService);
  const roleSrv = inject(RoleService);
  const router = inject(Router);

  const userRoles = await roleSrv.getRoles();
  //Get the last role which is Resource role, otherwise role service will send empy list
  const resourceAccessRole =  userRoles.at(-1) as string;
  if (resourceAccessRole.length) {
    return true;
  }
  //Otherwise clear the cache and redirect to forbidden page
  storageSrv.clearAll();
  router.navigate(['/forbidden']);
  return false;
};
