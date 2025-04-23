import { Injectable } from '@angular/core';
import { TrackingDetails, VerticalMenu, SubMenu } from '@ngtx-apps/data-access/models';
import { BehaviorSubject, Observable, of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { HttpClientService } from './common/http-client.service';
import * as menuItemsJson from '@assets/json/vertical-menu-data.json';
import { API_ENDPOINTS, environment } from '@ngtx-apps/utils/shared';

@Injectable({
  providedIn: 'root',
})
export class MenuService {

  private menuUrl = API_ENDPOINTS.userMeta.getMenu;
  //private menuJsonItems: VerticalMenu[] = (menuItemsJson as any).default; // URL to mock JSON file
  private menuItems!: VerticalMenu[];

  menuItemsSubject = new BehaviorSubject<VerticalMenu[]>([]);
  menuItems$ = this.menuItemsSubject.asObservable();

  constructor(private httpClientService: HttpClientService) { }

  /**
  * Call Menu API to fetch RBAC menu items
  */
  getMenuItems(roles: string[]): Observable<VerticalMenu[]> {
    if (environment.loadMenuFromService) {
      return this.httpClientService.get<any>(this.menuUrl).pipe(
        catchError(error => {
          console.error('Error fetching menu items', error);
          return of([]); // Return an empty array on error
        }),
        map(menuItems => {
          const filteredMenu = this.filterMenuItemsByRoles(menuItems, roles);
          this.menuItemsSubject.next(filteredMenu); // Update the subject
          return filteredMenu; // Return the filtered menu
        })
      );
    } else {
      // Mock JSON menu
      const localMenuItems = (menuItemsJson as any).default;
      console.log({localMenuItems, roles});
      const filteredMenu = this.filterMenuItemsByRoles(localMenuItems, roles);
      console.log({filteredMenu});
      this.menuItemsSubject.next(filteredMenu); // Update the subject
      return of(filteredMenu); // Return as observable
    }
  }


  /**
   * handle the role-based filtering of menu items.
   * @param menuItems
   * @param roles
   * @returns VerticalMenu[]
   */
  private filterMenuItemsByRoles(menuItems: VerticalMenu[], roles: string[]): VerticalMenu[] {
    // Filter menu items based on roles
    return menuItems.filter(item =>
      item.roles && item.roles.some(role => roles.includes(role))
    );
  }


  canAccessRoute(route: string): boolean {
    const menuItems = this.menuItemsSubject.getValue(); // Get current menu items

    // Function to recursively check if the route exists in the menu structure
    const checkRouteInMenu = (items: VerticalMenu[], route: string): boolean => {
      for (const item of items) {

        // Check if the current item's dataurl matches the route
        if (item.dataurl === route) {
          return true;
        }

        // If the item has submenus, recursively check them
        if (item.submenu) {
          const hasAccess = checkRouteInMenu(item.submenu, route);
          if (hasAccess) {
            return true;
          }
        }

        // Check if the item is a submenu and has items
        if (this.isSubMenu(item)) {
          const hasAccess = checkRouteInMenu(item.items, route);
          if (hasAccess) {
            return true;
          }
        }
      }
      return false; // Return false if the route is not found
    };

    return checkRouteInMenu(menuItems, route);
  }

  // Type guard to check if an item is a SubMenu
  private isSubMenu(item: VerticalMenu): item is SubMenu {
    return (item as SubMenu).items !== undefined;
  }

}
