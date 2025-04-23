import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
// import { AlertConstants } from '../../constants';
// import { AlertController, ConfirmController } from '../../models/common';

@Injectable({
  providedIn: 'root',
})
export class AlertBaseService {
  private alert$ = new BehaviorSubject<AlertController>({
    show: false,
    message: '',
  });
  alertController$ = this.alert$.asObservable();

  private confirm$ = new BehaviorSubject<ConfirmController>({
    show: false,
    message: '',
  });
  confirmController$ = this.confirm$.asObservable();


  /**
   * Method for showing an Alert
   * @param alertCtrl Object data for showing the Alert
   */
  openAlert(alertContrl: AlertController) {
    this.alert$.next({
      show: alertContrl.show,
      message: alertContrl.message,
      onClose: alertContrl.onClose ? alertContrl.onClose : null,
      title: alertContrl.title || AlertConstants.ALERT_TITLE,
      btnText: alertContrl.btnText || AlertConstants.ALERT_BTN_TEXT,
    });
  }

  closeAlert() {
    this.alert$.next({
      show: false,
      message: '',
      onClose: null,
    });
  }

  /**
   * Method for showing a Confirmation dialogue
   * @param confirmContrl Object data for showing the Confirm
   */
  openConfirm(confirmContrl: ConfirmController) {
    this.confirm$.next({
      show: confirmContrl.show,
      message: confirmContrl.message,
      onClose: confirmContrl.onClose ? confirmContrl.onClose : null,
      onAccept: confirmContrl.onAccept ? confirmContrl.onAccept : null,
      title: confirmContrl.title || AlertConstants.ALERT_TITLE,
      okBtnText:
        confirmContrl.okBtnText || AlertConstants.CONFIRM_ACCEPT_BTN_TEXT,
      closeBtnText:
        confirmContrl.closeBtnText || AlertConstants.CONFIRM_CANCEL_BTN_TEXT,
    });
  }

  closeConfirm() {
    this.confirm$.next({
      show: false,
      message: '',
      onClose: null,
      onAccept: null,
    });
  }
}
export const AlertConstants = {
  ALERT_TITLE: 'Alert',
  ALERT_BTN_TEXT: 'Okay',
  CONFIRM_TITLE: 'Confirm',
  CONFIRM_ACCEPT_BTN_TEXT: 'Accept',
  CONFIRM_CANCEL_BTN_TEXT: 'Cancel',
};


export interface AlertController {
  show: boolean;
  message: string;
  onClose?: any;
  title?: string;
  btnText?: string;
}

export interface ConfirmController {
  show: boolean;
  message: string;
  onClose?: any;
  onAccept?: any;
  title?: string;
  okBtnText?: string;
  closeBtnText?: string;
}
