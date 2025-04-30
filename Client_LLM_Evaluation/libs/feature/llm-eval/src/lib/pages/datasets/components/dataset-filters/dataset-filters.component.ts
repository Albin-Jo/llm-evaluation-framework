/* Path: libs/feature/llm-eval/src/lib/pages/datasets/components/dataset-filters/dataset-filters.component.ts */
import { Component, EventEmitter, Input, OnInit, Output, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';
import { DatasetFilterParams, DatasetStatus } from '@ngtx-apps/data-access/models';
import {
  QracButtonComponent,
  QracTextBoxComponent,
  QracSelectComponent
} from '@ngtx-apps/ui/components';

@Component({
  selector: 'app-dataset-filters',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    QracTextBoxComponent,
    QracSelectComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './dataset-filters.component.html',
  styleUrls: ['./dataset-filters.component.scss']
})
export class DatasetFiltersComponent implements OnInit {
  @Input() statusOptions: { value: string; label: string }[] = [];
  @Output() filterChange = new EventEmitter<Partial<DatasetFilterParams>>();
  @Output() searchChange = new EventEmitter<string>();

  filterForm: FormGroup;
  typeOptions = [
    { value: '', label: 'All Types' },
    { value: 'csv', label: 'CSV' },
    { value: 'jsonl', label: 'JSONL' },
    { value: 'txt', label: 'Text' },
    { value: 'custom', label: 'Custom' }
  ];

  visibilityOptions = [
    { value: 'true', label: 'Public' },
    { value: 'false', label: 'Private' },
    { value: '', label: 'All' }
  ];

  constructor(private fb: FormBuilder) {
    this.filterForm = this.fb.group({
      search: [''],
      status: [''],
      typeValue: [''],
      isPublic: ['true']
    });
  }

  ngOnInit(): void {
    // Set up search debounce
    this.filterForm.get('search')?.valueChanges
      .pipe(
        debounceTime(400),
        distinctUntilChanged()
      )
      .subscribe((value: string) => {
        this.searchChange.emit(value);
      });

    // Listen to other filter changes
    this.filterForm.get('status')?.valueChanges
      .subscribe((value: string) => {
        const status = value ? value as DatasetStatus : undefined;
        this.filterChange.emit({ status });
      });

    this.filterForm.get('typeValue')?.valueChanges
      .subscribe((value: string) => {
        // If the DatasetFilterParams interface doesn't have 'format', handle it as a custom property
        const filters: Partial<DatasetFilterParams> = {};
        // if (value) {
        //   // Map typeValue to type or handle as appropriate for your API
        //   filters.type = value;
        // }
        this.filterChange.emit(filters);
      });

    this.filterForm.get('isPublic')?.valueChanges
      .subscribe((value: string) => {
        const isPublic = value === 'true' ? true :
                        value === 'false' ? false : undefined;
        this.filterChange.emit({ is_public: isPublic });
      });
  }

  clearFilters(): void {
    this.filterForm.reset({
      search: '',
      status: '',
      typeValue: '',
      isPublic: 'true'
    });

    // Emit changes
    this.searchChange.emit('');
    this.filterChange.emit({
      status: undefined,
      is_public: true
    });
  }
}
