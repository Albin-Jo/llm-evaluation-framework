import { inject, Injectable } from '@angular/core';
import { Router } from '@angular/router';
import { StorageService } from './storage.service';
import { Idle, DEFAULT_INTERRUPTSOURCES } from '@ng-idle/core';
import { AuthenticationConstant, environment } from '@ngtx-apps/utils/shared';

@Injectable({
  providedIn: 'root',
})
export class IdleTimeoutService {

  private readonly router = inject(Router);
  private readonly storageService = inject(StorageService);
  private readonly idle = inject(Idle);

  isalreadyOpened: any = false;
  idleTimeOut = 0;
  countdown: any;

  public async subscribeIdletimeout() {

    this.idleTimeOut = environment.idleTime;
    this.idle.setIdle(this.idleTimeOut);
    // sets a timeout period of 5 seconds. after 10 seconds of inactivity, the user will be considered timed out.
    this.idle.setTimeout(this.idleTimeOut + 5);
    // sets the default interrupts, in this case, things like clicks, scrolls, touches to the document
    this.idle.setInterrupts(DEFAULT_INTERRUPTSOURCES);

    this.subsribeOntimeout();
    this.idle.onTimeoutWarning.subscribe(async (countdown: any) => {
      if (countdown < 30) {
        const sessionExpiryConfrm = confirm(
          AuthenticationConstant.extendSession +
          countdown +
          ' Second(s)! Click Okay to continue the session.'
        );

        if (sessionExpiryConfrm) {
          this.reset();
        } else {
          this.idle.stop();
          this.logout();
        }
      }
    });
    this.reset();
  }
  async reset() {
    const token = this.storageService.get(AuthenticationConstant.token);
    if (token != null) {
      this.idle.setIdle(this.idleTimeOut);
      this.idle.setTimeout(this.idleTimeOut + 5);
      this.idle.watch();
      this.isalreadyOpened = false;
    } else {
      console.log('After log out session triggered');
      this.countdown = null;
      this.idle.stop();
    }
  }
  subsribeOntimeout() {
    this.idle.onTimeout.subscribe(async () => {
      this.logout();
    });
  }
  async logout() {
    console.log('logout');
    this.storageService.clearAll();
    this.router.navigate(['/logout']);
  }

  public stopIdleTimeOut() {
    this.idle.stop();
  }
}
