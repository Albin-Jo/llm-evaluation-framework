import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter, OnInit, OnDestroy, AfterViewInit } from
'@angular/core';
import { FormControl, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
@Component({
  selector: 'qrac-textbox',
  templateUrl: './qractextbox.component.html',
  styleUrls: ['./qractextbox.component.scss'],
  imports: [CommonModule, ReactiveFormsModule]
})
export class QracTextBoxComponent implements OnInit, OnDestroy, AfterViewInit {

  @Input() label = '';
  @Input() inputValue = '';
  @Input() placeholder = 'Enter Value';
  @Input() isFloat = false;
  @Input() infoText= ''
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() errorMessage = 'This field is required';
  @Input() maxLengthErrorMessage = 'Max length is 100';
  @Input() initialValue = '';
  @Output() valueChange = new EventEmitter<string>();
  form!: FormGroup;
  control!: FormControl;
  ngOnInit() {
    this.control = new FormControl(this.initialValue, Validators.compose([this.required ? Validators.required : null, Validators.maxLength(100)]));
    this.form = new FormGroup({
      input: this.control
    });
  }

  ngAfterViewInit() {
    // Perform any post-initialization tasks here
  }

  ngOnDestroy() {
    // Cleanup tasks when the component is destroyed
  }

  get value() {
    return this.control!.value;
  }
  set value(val: string) {
    this.control!.setValue(val);
    this.valueChange.emit(val);
  }
 }
