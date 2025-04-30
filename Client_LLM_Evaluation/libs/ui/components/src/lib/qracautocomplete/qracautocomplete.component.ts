import { Component, OnInit, Input, Output, EventEmitter, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AutoCompleteModule } from '@syncfusion/ej2-angular-dropdowns'
@Component({
 selector: 'qrac-autocomplete',
 templateUrl: './qracautocomplete.component.html',
 styleUrls: ['./qracautocomplete.component.scss'],
 imports: [CommonModule, AutoCompleteModule],
})
export class QracAutocompleteComponent implements OnInit {
  constructor() {
    // Intentionally left empty
  }
  // defined the array of data
  public countriesData: { [key: string]: any }[] = [
   { Country: { Name: 'Australia' }, Code: { Id: 'AU' }},
      { Country: { Name: 'Bermuda' },Code: { Id: 'BM' }},
      { Country:{ Name: 'Canada'}, Code:{ Id: 'CA'} },
      { Country:{Name: 'Cameroon'}, Code:{ Id: 'CM'} },
      { Country:{Name: 'Denmark'}, Code:{ Id: 'DK' }},
      { Country:{Name: 'France'}, Code: { Id:'FR'} },
      { Country:{Name: 'Finland'}, Code:  { Id:'FI'} },
      { Country:{Name: 'Germany'}, Code: { Id:'DE'} },
      { Country:{Name: 'Greenland'}, Code:{ Id: 'GL' }},
      { Country:{Name: 'Hong Kong'}, Code: { Id:'HK'} },
      { Country:{Name: 'India'}, Code:{ Id: 'IN'} },
      { Country:{ Name: 'Italy'}, Code: { Id:'IT'} },
      { Country:{ Name: 'Japan'}, Code: { Id: 'JP'} },
      { Country:{Name: 'Mexico'}, Code: { Id: 'MX' }},
      { Country:{Name: 'Norway'}, Code: { Id: 'NO'} },
      { Country:{Name: 'Poland'}, Code: { Id: 'PL' }},
      { Country:{Name: 'Switzerland'}, Code: { Id: 'CH'} },
      { Country:{Name: 'United Kingdom'},Code: { Id: 'GB'} },
      { Country:{Name: 'United States'}, Code: { Id: 'US'} }
  ];
  // maps the appropriate column to fields property
  public fields: any = { value: 'Country.Name' };
  //set the placeholder to AutoComplete input
  public text = "Find a country";
  ngOnInit(): void {
    // any additional initialization
  }
  }
