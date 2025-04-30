import { Component } from '@angular/core';
import { AuthenticationService } from '@ngtx-apps/utils/services';
import { RouterLink, RouterLinkActive } from '@angular/router';
@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.scss'],
  imports: [RouterLink, RouterLinkActive]
})
export class HeaderComponent {

  constructor(private authService: AuthenticationService) { }

  logout() {
    this.authService.logout();
  }
}
