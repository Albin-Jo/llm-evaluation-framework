import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter, OnInit, OnDestroy, AfterViewInit } from
'@angular/core';
import { FormControl, FormGroup, Validators, ReactiveFormsModule, AbstractControl, ValidationErrors } from '@angular/forms';
@Component({
  selector: 'qrac-emailbox',
  templateUrl: './qracemailbox.component.html',
  styleUrls: ['./qracemailbox.component.scss'],
  imports: [CommonModule, ReactiveFormsModule]
})
export class QracEmailBoxComponent implements OnInit, OnDestroy, AfterViewInit {

  @Input() label = '';
  @Input() inputValue = '';
  @Input() placeholder = 'Enter Value';
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() errorMessage = 'This field is required';
  @Input() invalidEmailMessage = 'Please Enter a valid Email address';
  @Input() initialValue = '';
  @Input() isFloat = false;
  @Input() infoText= '';
  @Output() valueChange = new EventEmitter<string>();
  @Output() keyPressEvent = new EventEmitter<KeyboardEvent>();
  form!: FormGroup;
  control!: FormControl;

  validateEmailFormat(control: AbstractControl): ValidationErrors | null {
    const emailPattern = /^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,4}$/;
    return emailPattern.test(control.value) ? null : { 'email': true };
  }

  ngOnInit() {
    const validators = [this.validateEmailFormat.bind(this)];
    if (this.required) { validators.push(Validators.required); }
    this.control = new FormControl(this.inputValue, validators);
    this.form = new FormGroup({ input: this.control });
  }


  ngAfterViewInit() {
    // Perform any post-initialization tasks here
  }
  ngOnDestroy() {
    // Cleanup tasks when the component is destroyed
  }
  onKeyPress(event: KeyboardEvent) {
    this.keyPressEvent.emit(event);
  }
  get value() {
    return this.control!.value;
  }
  set value(val: string) {
    this.control!.setValue(val);
    this.valueChange.emit(val);
  }
 }
