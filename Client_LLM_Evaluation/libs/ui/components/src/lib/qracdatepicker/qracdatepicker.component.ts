import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { DatePickerModule } from '@syncfusion/ej2-angular-calendars'
import { FormControl, FormGroup, Validators, ReactiveFormsModule, ValidatorFn, AbstractControl, ValidationErrors } from '@angular/forms';
import { CommonModule } from '@angular/common';
@Component({
 selector: 'qrac-datepicker',
 templateUrl: './qracdatepicker.component.html',
 styleUrls: ['./qracdatepicker.component.scss'],
 imports: [CommonModule, DatePickerModule],
})
export class QracDatepickerComponent implements OnInit {
  @Input() label = 'Enter Date';
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() errorMessage = 'This field is required';
  @Input() disabled = false; // New input for disabling the checkboxes
  @Output() valueChange = new EventEmitter<any>();
  form!: FormGroup;
  public today: Date = new Date();
  public currentYear: number = this.today.getFullYear();
  public currentMonth: number = this.today.getMonth();
  public currentDay: number = this.today.getDate();
  public dateValue: any = new Date(new Date().setDate(14));
  public minDate: any = new Date(this.currentYear, this.currentMonth, 1);
  public maxDate: any =  new Date(this.currentYear, this.currentMonth, 27);
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

  set value(val: any) {
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
