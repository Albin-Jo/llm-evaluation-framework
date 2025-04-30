import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { DropDownListModule } from '@syncfusion/ej2-angular-dropdowns'
import { FilteringEventArgs } from '@syncfusion/ej2-dropdowns';
import { EmitType } from '@syncfusion/ej2-base';
import { CommonModule } from '@angular/common';
import { Query } from '@syncfusion/ej2-data';
import { FormControl, FormGroup, Validators, ReactiveFormsModule, ValidatorFn, AbstractControl, ValidationErrors } from '@angular/forms';

@Component({
 selector: 'qrac-dropdownlist',
 templateUrl: './qracdropdownlist.component.html',
 styleUrls: ['./qracdropdownlist.component.scss'],
 imports: [CommonModule, DropDownListModule],
})
export class QracDropdownlistComponent implements OnInit {
  @Input() label = 'Select option';
  @Input() required = false;
  @Input() isMandatory = true;
  @Input() data: { [key: string]: any }[] = [];
  @Input() errorMessage = 'Choose on item!';
  @Output() valueChange = new EventEmitter<string>();
  form!: FormGroup;
  control!: FormControl;

  constructor() {
    // Intentionally left empty
  }
      // defined the array of data
      //  public data: { [key: string]: any }[] = [
      //   { Id: "s3", Country: "Alaska" },
      //   { Id: "s1", Country: "California" },
      //   { Id: "s2", Country: "Florida" },
      //   { Id: "s4", Country: "Georgia" }];
    // maps the appropriate column to fields property
    public fields: any = { text: "Country", value: "Id" };
    // set the placeholder to the DropDownList input
    public text = "Select a country";
    //Bind the filter event
    public onFiltering: EmitType<FilteringEventArgs>  =  (e: FilteringEventArgs) => {
        let query = new Query();
        //frame the query based on search string with filter type.
        query = (e.text != "") ? query.where("Country", "startswith", e.text, true) : query;
        //pass the filter data source, filter query to updateData method.
        e.updateData(this.data, query);
    };

    ngOnInit() {
      this.control = new FormControl(this.data, Validators.required);
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
