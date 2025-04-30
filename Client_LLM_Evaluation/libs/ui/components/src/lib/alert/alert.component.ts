import { Component, OnInit } from '@angular/core';

import { NgStyle } from '@angular/common';
import { AlertBaseService } from '@ngtx-apps/utils/services';

@Component({
  selector: 'app-alert',
  templateUrl: './alert.component.html',
  styleUrls: ['./alert.component.scss'],
  imports: [NgStyle],
})
export class AlertComponent implements OnInit {
  showAlert: boolean;
  message: string;
  onCloseAlert: any;
  title: string | undefined;
  btnText: string | undefined;
  constructor(private alertBaseService: AlertBaseService) {
    this.showAlert = false;
    this.message = '';
    this.title = '';
    this.btnText = '';
  }

  ngOnInit(): void {
    this.alertBaseService.alertController$.subscribe((alertContrl) => {
      if (alertContrl) {
        this.showAlert = alertContrl.show;
        this.message = alertContrl.message;
        this.onCloseAlert = alertContrl.onClose;
        this.title = alertContrl.title;
        this.btnText = alertContrl.btnText;
      }
    });
  }

  async closeAlert() {
    if (this.onCloseAlert) {
      await this.onCloseAlert();
    }
    this.alertBaseService.closeAlert();
  }
}
