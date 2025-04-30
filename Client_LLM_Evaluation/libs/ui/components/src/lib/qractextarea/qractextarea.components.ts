import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter, OnInit, OnDestroy, AfterViewInit } from
'@angular/core';
import { FormControl, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
@Component({
  selector: 'qrac-textarea',
  templateUrl: './qractextarea.component.html',
  styleUrls: ['./qractextarea.component.scss'],
  imports: [CommonModule, ReactiveFormsModule]
})
export class QracTextAreaComponent implements OnInit, OnDestroy, AfterViewInit {

  @Input() label = '';
  @Input() inputValue = '';
  @Input() placeholder = 'Enter Value';
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() errorMessage = 'This field is required';
  @Input() maxLengthErrorMessage = 'Max length is 500';
  @Input() initialValue = '';
  @Input() isFloat = false;
  @Input() infoText= '';
  @Input() isMaxLimitRequired = true;
  @Input() maxLimit = 500;
  @Output() valueChange = new EventEmitter<string>();
  @Output() keyPressEvent = new EventEmitter<KeyboardEvent>();
  form!: FormGroup;
  control!: FormControl;
  remainingChars: number = this.maxLimit;

  ngOnInit() {
    const validators = [this.required ? Validators.required : null];
    if (this.isMaxLimitRequired) {
      validators.push(Validators.maxLength(this.maxLimit));
    }
    this.control = new FormControl(this.initialValue, Validators.compose(validators));
    this.form = new FormGroup({
      textarea: this.control
    });
    if (this.isMaxLimitRequired) {
      this.remainingChars = this.maxLimit - this.control.value.length;
      this.control.valueChanges.subscribe(value => {
        this.remainingChars = this.maxLimit - value.length;
      });
    }
  }

  ngAfterViewInit() {
    // Perform any post-initialization tasks here
  }
  ngOnDestroy() {
    // Cleanup tasks when the component is destroyed
  }

  onKeyPress(event: KeyboardEvent) {
    if (this.isMaxLimitRequired && this.control.value.length >= this.maxLimit) {
      event.preventDefault();
    } else {
      this.keyPressEvent.emit(event);
      if (this.isMaxLimitRequired) {
        this.remainingChars = this.maxLimit - this.control.value.length;
      }
    }
  }
  get value() {
    return this.control!.value;
  }
  set value(val: string) {
    this.control!.setValue(val);
    this.valueChange.emit(val);
  }
 }
