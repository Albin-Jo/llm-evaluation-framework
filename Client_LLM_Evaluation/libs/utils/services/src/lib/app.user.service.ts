import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { switchMap, map } from 'rxjs/operators';
import { RoleService } from './role.service';
import { StorageService } from './storage.service';

@Injectable({
  providedIn: 'root',
})
export class AppUserService {
  userRole = '';

  constructor(private storageSrv: StorageService, private roleSrv: RoleService) {}

  /**
   * Fetches user information and roles using Observables.
   */
  getUserInfo(): Observable<string[]> {
    return of(this.storageSrv.get('user')).pipe(
      map((user) => {
        try {
          return JSON.parse(user ?? '{}'); // Return an empty object if parsing fails
        } catch (error) {
          console.error('Error parsing user data:', error);
          return {};
        }
      }),
      switchMap(() => this.roleSrv.getRoles())
    );
  }
}
