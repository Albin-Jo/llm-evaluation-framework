import { Component, OnInit, Input, Output, EventEmitter, ViewChild } from '@angular/core';
import { UploaderModule } from '@syncfusion/ej2-angular-inputs'
import { CommonModule } from '@angular/common';
import { detach } from '@syncfusion/ej2-base';
import { UploaderComponent, FileInfo, SelectedEventArgs } from '@syncfusion/ej2-angular-inputs';
@Component({
 selector: 'qrac-upload',
 templateUrl: './qracupload.component.html',
 styleUrls: ['./qracupload.component.scss'],
 imports: [CommonModule, UploaderModule],
 providers: [UploaderComponent]
})
export class QracuploadComponent implements OnInit {
  @ViewChild('defaultupload')
  public uploadObj?: UploaderComponent;
  public path: any = {
    saveUrl: 'https://services.syncfusion.com/angular/production/api/FileUploader/Save',
    removeUrl: 'https://services.syncfusion.com/angular/production/api/FileUploader/Remove' };
  constructor() {
    // Intentionally left empty
  }
   ngAfterViewInit() {
    (document.getElementById('browse') as HTMLElement).onclick = function() {
    (document.getElementsByClassName('e-file-select-wrap')[0].querySelector('button') as HTMLButtonElement).click();
      return false;
    }
    document.getElementById('dropArea')!.onclick = (e: any) => {
          const target: HTMLElement = <HTMLElement>e.target;
          if (target.classList.contains('e-file-delete-btn')) {
              for (let i = 0; i < (this.uploadObj as UploaderComponent).getFilesData().length; i++) {
                  if ((target.closest('li') as HTMLLIElement).getAttribute('data-file-name') === (this.uploadObj as UploaderComponent).getFilesData()[i].name) {
                      (this.uploadObj as UploaderComponent).remove((this.uploadObj as UploaderComponent).getFilesData()[i]);
                  }
              }
          }
          else if (target.classList.contains('e-file-remove-btn')) {
              detach(target.closest('li') as HTMLLIElement);
          }
      }
  }
 public parentElement ?: HTMLElement;
  public progressbarContainer ?: HTMLElement;
  public filesDetails : FileInfo[] = [];
  public filesList: HTMLElement[] = [];
  public dropElement: HTMLElement = document.getElementsByClassName('control-fluid')[0] as HTMLElement;
  public onFileUpload(args: any) {
  const li: HTMLElement = (this.uploadObj as any)!.uploadWrapper?.querySelector('[data-file-name="' + args.file.name + '"]');
  const progressValue: number = Math.round((args.e.loaded / args.e.total) * 100);
  li.getElementsByTagName('progress')[0].value = progressValue;
  li.getElementsByClassName('percent')[0].textContent = progressValue.toString() + " %";
}
public onuploadSuccess(args: any) {
  if (args.operation === 'remove') {
      let height: string = document.getElementById('dropArea')!.style.height;
      height = (parseInt(height) - 40) + 'px';
      document.getElementById('dropArea')!.style.height = height;
  } else {
      const li: HTMLElement = (this.uploadObj as any).uploadWrapper.querySelector('[data-file-name="' + args.file.name + '"]');
      const progressBar: HTMLElement = li.getElementsByTagName('progress')[0];
      progressBar.classList.add('e-upload-success');
      li.getElementsByClassName('percent')[0].classList.add('e-upload-success');
      const height: string = document.getElementById('dropArea')!.style.height;
      document.getElementById('dropArea')!.style.height = parseInt(height) - 15 + 'px';
  }
}
public onuploadFailed(args: any) {
  const li: HTMLElement = (this.uploadObj as any).uploadWrapper.querySelector('[data-file-name="' + args.file.name + '"]');
  const progressBar: HTMLElement = li.getElementsByTagName('progress')[0];
  progressBar.classList.add('e-upload-failed');
  li.getElementsByClassName('percent')[0].classList.add('e-upload-failed');
}
public onSelect(args: SelectedEventArgs) {
  const length: number = args.filesData.length;
  let height: string = document.getElementById('dropArea')!.style.height;
  height = parseInt(height) + (length * 55) + 'px';
  document.getElementById('dropArea')!.style.height = height;
}
  ngOnInit(): void {
    // any additional initialization
  }
  }
