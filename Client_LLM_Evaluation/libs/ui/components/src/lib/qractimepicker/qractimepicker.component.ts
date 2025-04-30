import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { TimePickerModule } from '@syncfusion/ej2-angular-calendars'
import { FormControl, FormGroup, Validators, ReactiveFormsModule, ValidatorFn, AbstractControl, ValidationErrors } from '@angular/forms';
import { CommonModule } from '@angular/common';
@Component({
 selector: 'qrac-timepicker',
 templateUrl: './qractimepicker.component.html',
 styleUrls: ['./qractimepicker.component.scss'],
 imports: [CommonModule, TimePickerModule],
})
export class QracTimepickerComponent implements OnInit {
  @Input() label = 'Enter Date';
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() errorMessage = 'This field is required';
  @Input() disabled = false; // New input for disabling the checkboxes
  @Output() valueChange = new EventEmitter<string>();
  form!: FormGroup;
  public month: number = new Date().getMonth();
  public fullYear: number = new Date().getFullYear();
  public date: number = new Date().getDate();
  public dateValue: Date = new Date(this.fullYear, this.month , this.date, 10, 0, 0);
  public minValue: Date = new Date(this.fullYear, this.month , this.date, 7, 0, 0);
  public maxValue: Date = new Date(this.fullYear, this.month, this.date, 16, 0 ,0);

  control!: FormControl;

  constructor() {
    // Intentionally left empty
  }
  ngOnInit() {
    this.control = new FormControl(this.dateValue, Validators.required);
  }

  get value() {
    return this.control!.value;
  }

  set value(val: string) {
    if (val === null && this.required) {
      this.control.markAsDirty();
      this.control.markAsTouched();
      this.control.setErrors({ 'required': true });
    } else {
      this.control.setValue(val);
      this.valueChange.emit(val);
    }
  }
}
