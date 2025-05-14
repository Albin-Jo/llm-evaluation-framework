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
    QracSelectComponent,
    QracButtonComponent
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './dataset-filters.component.html',
  styleUrls: ['./dataset-filters.component.scss']
})
export class DatasetFiltersComponent implements OnInit {
  @Input() statusOptions: { value: string; label: string }[] = [
    { value: '', label: 'All Statuses' },
    { value: DatasetStatus.READY, label: 'Ready' },
    { value: DatasetStatus.PROCESSING, label: 'Processing' },
    { value: DatasetStatus.ERROR, label: 'Error' }
  ];

  @Input() typeOptions: { value: string; label: string }[] = [
    { value: '', label: 'All Types' },
    { value: 'question_answer', label: 'Question Answer' },
    { value: 'user_query', label: 'User Query' },
    { value: 'reference', label: 'Reference' },
    { value: 'evaluation', label: 'Evaluation' },
    { value: 'custom', label: 'Custom' }
  ];

  @Input() visibilityOptions: { value: string; label: string }[] = [
    { value: 'true', label: 'Public' },
    { value: 'false', label: 'Private' },
    { value: '', label: 'All' }
  ];

  @Input() dateRangeOptions: { value: string; label: string }[] = [
    { value: '', label: 'Any Time' },
    { value: 'today', label: 'Today' },
    { value: 'yesterday', label: 'Yesterday' },
    { value: 'week', label: 'This Week' },
    { value: 'month', label: 'This Month' }
  ];

  @Input() sizeRangeOptions: { value: string; label: string }[] = [
    { value: '', label: 'Any Size' },
    { value: 'small', label: 'Small (<1MB)' },
    { value: 'medium', label: 'Medium (1-10MB)' },
    { value: 'large', label: 'Large (>10MB)' }
  ];

  @Output() filterChange = new EventEmitter<Partial<DatasetFilterParams>>();
  @Output() searchChange = new EventEmitter<string>();
  @Output() clearFiltersEvent = new EventEmitter<void>();

  filterForm: FormGroup;

  constructor(private fb: FormBuilder) {
    this.filterForm = this.fb.group({
      search: [''],
      status: [''],
      type: [''],
      isPublic: ['true'],
      dateRange: [''],
      sizeRange: ['']
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

    // Listen to status changes
    this.filterForm.get('status')?.valueChanges
      .subscribe((value: string) => {
        const status = value ? value as DatasetStatus : undefined;
        this.filterChange.emit({ status });
      });

    // Listen to type changes
    this.filterForm.get('type')?.valueChanges
      .subscribe((value: string) => {
        this.filterChange.emit({ type: value || undefined });
      });

    // Listen to visibility changes
    this.filterForm.get('isPublic')?.valueChanges
      .subscribe((value: string) => {
        const isPublic = value === 'true' ? true :
                        value === 'false' ? false : undefined;
        this.filterChange.emit({ is_public: isPublic });
      });

    // Listen to date range changes
    this.filterForm.get('dateRange')?.valueChanges
      .subscribe((value: string) => {
        this.updateDateRangeFilter(value);
      });

    // Listen to size range changes
    this.filterForm.get('sizeRange')?.valueChanges
      .subscribe((value: string) => {
        this.updateSizeRangeFilter(value);
      });
  }

  private updateDateRangeFilter(value: string): void {
    const dateFilter: Partial<DatasetFilterParams> = {};

    if (value) {
      const now = new Date();
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());

      switch (value) {
        case 'today':
          dateFilter.dateFrom = today.toISOString();
          break;
        case 'yesterday':
          const yesterday = new Date(today);
          yesterday.setDate(yesterday.getDate() - 1);
          dateFilter.dateFrom = yesterday.toISOString();
          dateFilter.dateTo = today.toISOString();
          break;
        case 'week':
          const weekStart = new Date(today);
          weekStart.setDate(weekStart.getDate() - weekStart.getDay());
          dateFilter.dateFrom = weekStart.toISOString();
          break;
        case 'month':
          const monthStart = new Date(today.getFullYear(), today.getMonth(), 1);
          dateFilter.dateFrom = monthStart.toISOString();
          break;
      }
    } else {
      dateFilter.dateFrom = undefined;
      dateFilter.dateTo = undefined;
    }

    this.filterChange.emit(dateFilter);
  }

  private updateSizeRangeFilter(value: string): void {
    const sizeFilter: Partial<DatasetFilterParams> = {};

    if (value) {
      switch (value) {
        case 'small':
          sizeFilter.sizeMax = 1024 * 1024; // 1MB
          break;
        case 'medium':
          sizeFilter.sizeMin = 1024 * 1024; // 1MB
          sizeFilter.sizeMax = 10 * 1024 * 1024; // 10MB
          break;
        case 'large':
          sizeFilter.sizeMin = 10 * 1024 * 1024; // 10MB
          break;
      }
    } else {
      sizeFilter.sizeMin = undefined;
      sizeFilter.sizeMax = undefined;
    }

    this.filterChange.emit(sizeFilter);
  }

  clearFilters(): void {
    this.filterForm.reset({
      search: '',
      status: '',
      type: '',
      isPublic: 'true',
      dateRange: '',
      sizeRange: ''
    });

    // Emit base filter values to reset everything
    this.searchChange.emit('');
    this.filterChange.emit({
      status: undefined,
      type: undefined,
      is_public: true,
      dateFrom: undefined,
      dateTo: undefined,
      sizeMin: undefined,
      sizeMax: undefined
    });

    // Emit a separate event for parent component to handle complete reset if needed
    this.clearFiltersEvent.emit();
  }
}