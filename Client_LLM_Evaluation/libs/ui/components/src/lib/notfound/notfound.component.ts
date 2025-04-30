import { CUSTOM_ELEMENTS_SCHEMA, Component, ElementRef, ViewChild, inject } from '@angular/core';
import { Router } from '@angular/router';
import { PATHS } from '@ngtx-apps/utils/shared';
@Component({
  selector: 'ex-notfound',
  templateUrl: './notfound.component.html',
  styleUrls: ['./notfound.component.scss'],
  schemas:[CUSTOM_ELEMENTS_SCHEMA]
})
export class NotfoundComponent {
  @ViewChild('SideModalComponent', { read: ElementRef, static: false }) SideModalComponent!: ElementRef;

  private readonly router = inject(Router);

  buttonClicked() {
    this.router.navigate([PATHS.APP]);
  }

}
