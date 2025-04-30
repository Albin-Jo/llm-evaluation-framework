import { CommonModule } from '@angular/common';
import { Component, CUSTOM_ELEMENTS_SCHEMA, DestroyRef, ElementRef, inject, OnInit, ViewChild } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRoute, NavigationEnd, Router, RouterOutlet } from '@angular/router';
import { VerticalMenu } from '@ngtx-apps/data-access/models';
import { MenuService } from '@ngtx-apps/data-access/services';
import { QracButtonComponent, QrecwebheaderComponent } from '@ngtx-apps/ui/components';
import { AppUserService, RoleService, StorageService } from '@ngtx-apps/utils/services';
import { environment, FEATURE_PATHS, PATHS } from '@ngtx-apps/utils/shared';
import { GridModule } from '@syncfusion/ej2-angular-grids';
import { Observable } from 'rxjs';
import { filter, switchMap } from 'rxjs/operators';
import { PageTab } from './pagetabs.interface';

@Component({
  selector: 'app-layout',
  templateUrl: './layout.page.html',
  styleUrls: ['./layout.page.scss'],
  imports: [
    CommonModule,
    RouterOutlet,
    QrecwebheaderComponent,
    GridModule,
    QracButtonComponent
],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class LayoutPage implements OnInit {
  environment = environment;
  private readonly storageSrv = inject(StorageService);
  private readonly roleSrv = inject(RoleService);
  storedUser: any;
  userResourceAccessRole!: string;

  @ViewChild('verticalmenu', { read: ElementRef, static: false }) verticalmenu!: ElementRef;
  @ViewChild('SideModalComponent', { read: ElementRef, static: false }) SideModalComponent!: ElementRef;

  public menuItems!: VerticalMenu[];
  userimgdata = 'assets/images/common/profile.svg';
  dropdownValues = ['Option 1', 'Option 2', 'Option 3'];
  pagetabs: PageTab[] = [];
  filtervaluesData = [
    { "id": 1, "name": "Option 1" },
    { "id": 2, "name": "Option 2" },
    { "id": 3, "name": "Option 3" }
  ];

  constructor(
    private router: Router,
    private activatedRoute: ActivatedRoute, private menuSrv: MenuService,
    private appUserService: AppUserService, private destroyRef: DestroyRef
  ) { }


  ngOnInit(): void {
    //Router event required in case screen refreshed and user will redirect to default tab
    this.router.events
      .pipe(
        filter((event) => event instanceof NavigationEnd),
        switchMap(() => this.fetchUserAndMenuItems()),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe({
        next: (items) => this.handleMenuItems(items),
        error: (err) => console.error('Error fetching user info or menu items:', err)
      });

    this.fetchUserAndMenuItems()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (items) => this.handleMenuItems(items),
        error: (err) => console.error('Error fetching user info or menu items:', err)
      });
  }


  fetchUserAndMenuItems(): Observable<any> {
    return this.appUserService.getUserInfo().pipe(
      switchMap((role) => {
        //Last role assumes as access role to be shown in user menu
        this.userResourceAccessRole = role[role.length - 1] as string;
        this.storedUser = JSON.parse(localStorage.getItem('user') ?? '');
        return this.menuSrv.getMenuItems(role);
      })
    );
  }

  handleMenuItems(items: any): void {
    if (items) {
      this.menuItems = items;
      const url = this.router.url;
      this.addTabForUrl(url);
    }
  }


  /**
   * Adds or activates a tab based on the provided URL.
   * Extracts the main menu and submenu IDs from the URL, finds the corresponding menu items,
   * and adds a new tab if it doesn't already exist.
   *
   * @param {string} url - The URL from which to extract menu and submenu IDs.
   * @returns {void}
   */
  addTabForUrl(url: string): void {
    let [mainMenuId, submenuId] = url.split('/').slice(-2); // Extract last two segments
    let mainMenuAppId = (mainMenuId==PATHS.APP) ? submenuId : mainMenuId;
    let mainMenuItem = this.menuItems.find((item: VerticalMenu) => item.dataurl === mainMenuAppId);

    if (!mainMenuItem) {
      console.log('Main menu item not found:', mainMenuAppId);
      return;
    }

    if (mainMenuItem.name && mainMenuItem.dataurl && mainMenuId==PATHS.APP) {
      this.addOrActivateTab(mainMenuItem.name, mainMenuItem.dataurl);
    }

    const gluedSubMenu = mainMenuAppId+'/'+submenuId;

    const submenuItem = mainMenuItem.submenu?.flatMap(sub => sub.items).find(item => item.dataurl === gluedSubMenu);
    if (!submenuItem) {
      console.log('Submenu item not found:', submenuId);
      return;
    }

    // Ensure submenuItem.name and submenuItem.dataurl are defined
    if (submenuItem.name && submenuItem.dataurl) {
      this.addOrActivateTab(submenuItem.name, submenuItem.dataurl);
    } else {
      console.log('Submenu item name or dataurl is undefined:', submenuItem);
    }
  }

  /**
  * Handles the menu selection event. Adds or activates a tab based on the selected menu item
  * and navigates to the corresponding URL.
  *
  * @param {Event} event - The menu selection event containing the selected item's details.
  * @returns {void}
  */
  handleMenuSelect(event: Event): void {
    const { name, dataurl } = (event as CustomEvent).detail;
    this.addOrActivateTab(name, dataurl);
     this.router.navigate([PATHS.APP +'/' + dataurl]);
  }

  /**
  * Adds a new tab or activates an existing tab based on the provided title and data URL.
  * If the tab doesn't exist, it creates a new one and adds it to the `pagetabs` array.
  * If the tab already exists, it sets it as the active tab.
  *
  * @param {string} title - The title of the tab.
  * @param {string} dataurl - The data URL associated with the tab.
  * @returns {void}
  */
  addOrActivateTab(title: string, dataurl: string): void {
    const existingTab = this.pagetabs.find(tab => tab.dataurl === dataurl);
    let closable = (dataurl!==FEATURE_PATHS.HOME);
    if (!existingTab) {
      this.pagetabs.push({ title, active: true, closable: closable, icon: '', dataurl });
    }

    this.pagetabs.forEach(tab => (tab.active = tab.dataurl === dataurl));
  }

  onClickProfile() {
    const title = "Profile";
    const url = FEATURE_PATHS.PROFILE;
    const event = new CustomEvent(title, { detail: {name: title, dataurl: url} });
    this.handleMenuSelect(event);
  }
  onClickLogOut() {
    console.log('logging out');
    this.storageSrv.clearAll();
    this.router.navigate([PATHS.LOGOUT]);
  }
  openmegamenu() {
    alert();
  }
  opennotify() {
    alert();
  }
  handleFormSubmit(event: { dropdownValue: string; textboxValue: string }) {
    console.log('Dropdown Value:', event.dropdownValue);
    console.log('Textbox Value:', event.textboxValue);
  }

  closeModal1() {
    this.SideModalComponent.nativeElement.closeModal();
  }
  openModal1() {
    this.SideModalComponent.nativeElement.openModal();
  }

}
