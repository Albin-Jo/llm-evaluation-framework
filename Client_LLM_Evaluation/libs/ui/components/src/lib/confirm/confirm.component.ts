import { Component, OnInit } from '@angular/core';
import { AlertBaseService } from '@ngtx-apps/utils/services';
import { NgStyle } from '@angular/common';

@Component({
  selector: 'app-confirm',
  templateUrl: './confirm.component.html',
  styleUrls: ['./confirm.component.scss'],
  imports: [NgStyle],
})
export class ConfirmComponent implements OnInit {
  showConfirm: boolean;
  message: string;
  onCloseCallback: any;
  onAcceptCallback: any;
  title: string | undefined;
  okBtnText: string | undefined;
  closeBtnText: string | undefined;
  constructor(private alertBaseService: AlertBaseService) {
    this.showConfirm = false;
    this.message = '';
    this.title = '';
    this.okBtnText = '';
    this.closeBtnText = '';
  }

  ngOnInit(): void {
    this.alertBaseService.confirmController$.subscribe((confirmContrl: any) => {
      if (confirmContrl) {
        this.showConfirm = confirmContrl.show;
        this.message = confirmContrl.message;
        this.onCloseCallback = confirmContrl.onClose;
        this.onAcceptCallback = confirmContrl.onAccept;
        this.title = confirmContrl.title;
        this.okBtnText = confirmContrl.okBtnText;
        this.closeBtnText = confirmContrl.closeBtnText;
      }
    });
  }

  async closeConfrim() {
    if (this.onCloseCallback) {
      await this.onCloseCallback();
    }
    this.alertBaseService.closeConfirm();
  }

  async acceptConfirm() {
    if (this.onAcceptCallback) {
      await this.onAcceptCallback();
    }
    this.alertBaseService.closeConfirm();
  }
}
