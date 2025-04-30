import { inject, Injectable } from '@angular/core';
import { AlertBaseService, AlertController, ConfirmController } from './alertBase.service';

@Injectable({
  providedIn: 'root',
})
export class AlertService {
  private readonly alertBaseService = inject(AlertBaseService);

  showAlert(alertCtrl: AlertController) {
    this.alertBaseService.openAlert(alertCtrl);
  }

  showConfirm(confrmCtrl: ConfirmController) {
    this.alertBaseService.openConfirm(confrmCtrl);
  }
}
