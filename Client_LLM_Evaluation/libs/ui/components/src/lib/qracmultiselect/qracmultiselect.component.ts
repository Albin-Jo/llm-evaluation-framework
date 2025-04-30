import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { MultiSelectModule, FilteringEventArgs  } from '@syncfusion/ej2-angular-dropdowns'
import { FormControl, FormGroup, Validators, ReactiveFormsModule, ValidatorFn, AbstractControl, ValidationErrors } from '@angular/forms';
import { DataManager,Query } from '@syncfusion/ej2-data';
import { EmitType } from '@syncfusion/ej2-base';
import { CommonModule } from '@angular/common';
@Component({
 selector: 'qrac-multiselect',
 templateUrl: './qracmultiselect.component.html',
 styleUrls: ['./qracmultiselect.component.scss'],
 imports: [CommonModule, MultiSelectModule],
})
export class QracMultiselectComponent implements OnInit {
  @Input() label = 'Select options';
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() data: { [key: string]: string }[] = [];
  @Input() errorMessage = 'At least one selected data is required.';
  @Output() valueChange = new EventEmitter<string[]>();
  form!: FormGroup;
  control!: FormControl;

  // map the appropriate column
  public fields: any = { text: "country", value: "index" };
  // set the placeholder to the MultiSelect input
  public placeholder = 'Select countries';
  //Bind the filter event
  public onFiltering: EmitType<FilteringEventArgs>  =  (e: FilteringEventArgs) => {
      let query = new Query();
      //frame the query based on search string with filter type.
      query = (e.text != "") ? query.where("country", "startswith", e.text, true) : query;
      //pass the filter data source, filter query to updateData method.
      e.updateData(this.data, query);
  };

  constructor() {
    // Intentionally left empty
  }

  atLeastOneOptionSelectedValidator(): ValidatorFn {
    return (control: AbstractControl): ValidationErrors | null => {
      const value = control.value;
      if (Array.isArray(value) && value.length > 0) {
        return null;  // no error
      } else {
        return { 'required': true };  // error: at least one option must be selected
      }
    };
  }
  ngOnInit() {
    this.control = new FormControl(this.data, this.atLeastOneOptionSelectedValidator());  }

  get value() {
    return this.control!.value;
  }


  set value(val: string[]) {
    if (val.length === 0 && this.required) {
      this.control.markAsDirty();
      this.control.markAsTouched();
      this.control.setErrors({ 'required': true });
    } else {
      this.control.setValue(val);
      this.valueChange.emit(val);
    }
  }
  }
