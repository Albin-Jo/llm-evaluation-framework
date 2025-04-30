/* Path: libs/feature/llm-eval/src/lib/pages/prompts/components/prompt-filters/prompt-filters.component.ts */
import { Component, EventEmitter, Input, OnInit, Output, NO_ERRORS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';
import { PromptFilter } from '@ngtx-apps/data-access/models';
import { QracSelectComponent } from '@ngtx-apps/ui/components';

// Extended filter interface to include additional filter properties
interface ExtendedPromptFilter extends PromptFilter {
  search?: string;
  sort_by?: string;
}

@Component({
  selector: 'app-prompt-filters',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule
  ],
  schemas: [NO_ERRORS_SCHEMA],
  templateUrl: './prompt-filters.component.html',
  styleUrls: ['./prompt-filters.component.scss']
})
export class PromptFiltersComponent implements OnInit {
  @Input() currentFilter: ExtendedPromptFilter = {};
  @Output() filterChange = new EventEmitter<ExtendedPromptFilter>();

  filterForm!: FormGroup;

  constructor(private fb: FormBuilder) {}

  ngOnInit(): void {
    this.initializeForm();
    this.setupFormListeners();
  }

  private initializeForm(): void {
    this.filterForm = this.fb.group({
      is_public: [this.currentFilter.is_public ?? null],
      template_id: [this.currentFilter.template_id ?? null],
      search: [this.currentFilter.search ?? ''],
      sort_by: [this.currentFilter.sort_by ?? 'updated_at']
    });
  }

  private setupFormListeners(): void {
    // Listen for form changes with debounce
    this.filterForm.valueChanges
      .pipe(
        debounceTime(300),
        distinctUntilChanged((prev, curr) => {
          return JSON.stringify(prev) === JSON.stringify(curr);
        })
      )
      .subscribe(value => {
        const filters: ExtendedPromptFilter = {};

        if (value.search && value.search.trim() !== '') {
          filters.search = value.search.trim();
        }

        if (value.is_public !== null && value.is_public !== undefined) {
          filters.is_public = value.is_public;
        }

        if (value.template_id) {
          filters.template_id = value.template_id;
        }

        if (value.sort_by) {
          filters.sort_by = value.sort_by;
        }

        // Reset pagination when filters change
        filters.skip = 0;
        // Keep the current limit
        filters.limit = this.currentFilter.limit;

        this.filterChange.emit(filters);
      });
  }

  /**
   * Reset filters to default values
   */
  resetFilters(): void {
    this.filterForm.patchValue({
      is_public: null,
      template_id: null,
      search: '',
      sort_by: 'updated_at'
    });
  }

  /**
   * Toggle between public, private, and all prompts
   */
  toggleVisibility(): void {
    const currentValue = this.filterForm.get('is_public')?.value;
    // Cycle through: null (all) -> true (public) -> false (private) -> null (all)
    let newValue = null;

    if (currentValue === null) {
      newValue = true;
    } else if (currentValue === true) {
      newValue = false;
    }

    this.filterForm.patchValue({ is_public: newValue });
  }

  /**
   * Get display value for visibility button
   */
  getVisibilityLabel(): string {
    const currentValue = this.filterForm.get('is_public')?.value;
    if (currentValue === true) return 'Public';
    if (currentValue === false) return 'Private';
    return 'All';
  }
}
