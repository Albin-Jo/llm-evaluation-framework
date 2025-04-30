import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter, OnInit, OnDestroy, AfterViewInit, ElementRef, Renderer2, HostListener } from
  '@angular/core';
import { FormControl, FormGroup, Validators, ReactiveFormsModule, FormsModule } from '@angular/forms';
@Component({
  selector: 'qrac-numberinput',
  templateUrl: './qracnumberinput.component.html',
  styleUrls: ['./qracnumberinput.component.scss'],
  imports: [CommonModule, ReactiveFormsModule, FormsModule]
})
export class QracNumberInputComponent implements OnInit, OnDestroy, AfterViewInit {

  @Input() label = '';
  @Input() inputValue = '';
  @Input() placeholder = 'Enter Value';
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() errorMessage = 'This field is required';
  @Input() initialValue = '';
  @Input() isFloat = false;
  @Input() infoText = '';
  @Input() countryCodes: { code: string, name: string }[] = [];
  @Input() isCountryCode = false;
  @Output() valueChange = new EventEmitter<string>();
  form!: FormGroup;
  control!: FormControl;
  countryCodeControl!: FormControl;
  selectedCountryCode = '+974';

  ngOnInit() {
    this.control = new FormControl(this.initialValue, Validators.compose([this.required ? Validators.required : null, Validators.pattern('\\+?[0-9]*')]));
    this.countryCodeControl = new FormControl('', Validators.required);
    this.form = new FormGroup({
      countryCode: this.countryCodeControl,
      input: this.control
    });
  }

  constructor(private el: ElementRef, private renderer: Renderer2) {
    // Intentionally left empty
  }

  ngAfterViewInit() {
    this.onSelectChange({} as Event)
  }

  @HostListener('change', ['$event'])
  onSelectChange(event: Event) {
    // Get the .custom-select and .phonenumber-input elements
    const selectElement = this.el.nativeElement.querySelector('.custom-select');
    const inputElement = this.el.nativeElement.querySelector('.phonenumber-input');

    // Calculate the padding (in pixels)
    const padding = (selectElement?.offsetWidth || '0') + 'px';

    // Set the padding-left of .phonenumber-input
    this.renderer.setStyle(inputElement, 'padding-left', padding);
  }
  ngOnDestroy() {
    // Cleanup tasks when the component is destroyed
  }

  updateSelectedCode(event: Event): void {
    const selectElement = event.target as HTMLSelectElement;
    this.selectedCountryCode = selectElement.value;
  }

  onKeyDown(event: any) {
    const pattern = /[0-9+]/; // Allow numbers and the "+" sign
    const inputChar = event.key;

    if (!pattern.test(inputChar) && inputChar !== "Backspace" && inputChar !== "Enter") {
      // Invalid character and not a backspace, prevent input
      event.preventDefault();
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
