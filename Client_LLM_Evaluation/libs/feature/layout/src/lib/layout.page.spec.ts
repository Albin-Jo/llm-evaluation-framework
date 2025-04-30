import { ComponentFixture, TestBed } from '@angular/core/testing';
import { LayoutPage } from './layout.page';
import { AppUserService, StorageService } from '@ngtx-apps/utils/services';
import { MenuService } from '@ngtx-apps/data-access/services';
import { of } from 'rxjs';
import { RouterTestingModule } from '@angular/router/testing';
import { By } from '@angular/platform-browser';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { platformBrowserDynamicTesting as angularPlatformBrowserDynamicTesting, BrowserDynamicTestingModule } from '@angular/platform-browser-dynamic/testing';

describe('LayoutPage', () => {
  let component: LayoutPage;
  let fixture: ComponentFixture<LayoutPage>;
  let menuServiceMock: jest.Mocked<MenuService>;
  let appUserServiceMock: jest.Mocked<AppUserService>;
  let storageServiceMock: jest.Mocked<StorageService>;
  let router: Router;
  const mockMenuResponse = [{ id: '1', name: 'Home', dataurl: '/home' }];
  const mockUser = 'admin';
  const mockStoredUser = { name: 'John Doe', preferred_username: 'johndoe' };

  beforeEach(async () => {
    menuServiceMock = {
      getMenuItems: jest.fn().mockReturnValue(of(mockMenuResponse))
    } as any;

    appUserServiceMock = {
      getUserInfo: jest.fn().mockReturnValue(of(mockUser))
    } as any;


    storageServiceMock = {
      get: jest.fn().mockReturnValue(JSON.stringify(mockStoredUser)) // Mock valid JSON data
    } as any;

    await TestBed.configureTestingModule({
      imports: [RouterTestingModule, CommonModule, LayoutPage],
      providers: [
        { provide: MenuService, useValue: menuServiceMock },
        { provide: AppUserService, useValue: appUserServiceMock },
        { provide: StorageService, useValue: storageServiceMock }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(LayoutPage);
    component = fixture.componentInstance;
    router = TestBed.inject(Router);

    // Set the conditions for the deferred content
    component.storedUser = mockStoredUser;
    component.userResourceAccessRole = mockUser;
    component.pagetabs = [];
    component.menuItems = mockMenuResponse;

    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should render deferred content when conditions are met', async () => {


    // Trigger change detection to render the deferred content
    fixture.detectChanges();
    await fixture.whenStable(); // Wait for asynchronous operations to complete
  fixture.detectChanges(); // Trigger change detection again
    // Check if the deferred content is rendered
    const webHeaderElement = fixture.nativeElement.querySelector('qrds-webheader');
    const verticalMenuElement = fixture.nativeElement.querySelector('qrsc-verticalmenu');

    expect(webHeaderElement).toBeTruthy();
    expect(verticalMenuElement).toBeTruthy();
  });

  xit('should render the menu after initialization', async () => {
    // Wait for asynchronous operations to complete
    fixture.detectChanges();
    await fixture.whenStable();

    // Ensure the menu service was called
    expect(menuServiceMock.getMenuItems).toHaveBeenCalled();

    // Check if the menu items are rendered
    const menuElement = fixture.debugElement.query(By.css('qrsc-verticalmenu'));
    expect(menuElement).toBeTruthy();

    // Check if the correct menu items are passed to the component
    expect(component.menuItems).toEqual(mockMenuResponse);
  });

  it('should fetch user info', async () => {
    // Wait for asynchronous operations to complete
    fixture.detectChanges();
    await fixture.whenStable();

    // Verify that the methods were called
    expect(appUserServiceMock.getUserInfo).toHaveBeenCalled();

    // Verify that the stored user is set correctly
    expect(component.storedUser).toEqual(mockStoredUser);
  });

  it('should add a tab when a menu item is selected', () => {
    const mockEvent = {
      detail: {
        name: 'Home',
        dataurl: '/home'
      }
    };

    // Trigger the menu select event
    component.handleMenuSelect(mockEvent as any);

    // Check if the tab was added
    expect(component.pagetabs.length).toBe(1);
    expect(component.pagetabs[0]).toEqual({
      title: 'Home',
      active: true,
      closable: true,
      icon: '',
      dataurl: '/home'
    });
  });

  it('should activate the correct tab when navigating to a URL', () => {
    // Set up the menu items
    component.menuItems = [
      {
        id: 'pricing',
        submenu: [
          {
            items: [
              { id: 'sub1', name: 'Sub 1', dataurl: '/pricing/sub1' },
              { id: 'sub2', name: 'Sub 2', dataurl: '/pricing/sub2' }
            ]
          }
        ]
      }
    ];

    // Simulate navigating to a URL
    component.addTabForUrl('/pricing/sub1');
    fixture.detectChanges();

    // Check if the correct tab is activated
    const activeTab = component.pagetabs.find((tab: any) => tab.active);
    expect(activeTab).toBeTruthy();
    expect(activeTab?.dataurl).toBe('/pricing/sub1');
  });
});
