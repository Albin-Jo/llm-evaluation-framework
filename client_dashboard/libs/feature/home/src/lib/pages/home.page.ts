/* Path: libs/feature/home/src/lib/pages/home.page.ts */
import { CommonModule, NgFor, NgIf } from '@angular/common';
import { CUSTOM_ELEMENTS_SCHEMA, NO_ERRORS_SCHEMA } from '@angular/core';
import { AfterViewInit, Component, ElementRef, inject, OnInit, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { FilterService, FreezeService, GridModule, SortService } from '@syncfusion/ej2-angular-grids';

import {
  QracButtonComponent,
  QracuploadComponent
} from '@ngtx-apps/ui/components';
import {
  DatasetService,
  PromptService
} from '@ngtx-apps/data-access/services';
import { IdleTimeoutService } from '@ngtx-apps/utils/services';
import {
  Dataset,
  DatasetStatus,
  PromptResponse
} from '@ngtx-apps/data-access/models';

interface EvaluationSummary {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed' | 'scheduled';
  datasetName: string;
  promptName: string;
  modelCount: number;
  avgRelevance?: number;
  avgLatency?: number;
  createdAt: string;
}

@Component({
  selector: 'app-home',
  templateUrl: './home.page.html',
  styleUrls: ['./home.page.scss'],
  imports: [
    CommonModule,
    NgFor,
    FormsModule,
    GridModule,
    NgIf,
    QracButtonComponent,
    ReactiveFormsModule,
    RouterModule
  ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA, NO_ERRORS_SCHEMA],
  providers: [SortService, FilterService, FreezeService]
})
export class HomePage implements OnInit, AfterViewInit {

  // Modal references
  @ViewChild('qrscsystoast', { read: ElementRef, static: false }) qrscsystoast!: ElementRef;

  // Dashboard data
  activeTab = 'datasets';
  isLoading = false;

  // Feature stats
  datasetsCount = 0;
  promptsCount = 0;
  evaluationsCount = 0;
  reportsCount = 0;

  // Dashboard data
  datasets: Dataset[] = [];
  recentPrompts: PromptResponse[] = [];
  recentEvaluations: EvaluationSummary[] = [];

  // Status enum for template access
  datasetStatus = DatasetStatus;

  private readonly idleTimeoutService = inject(IdleTimeoutService);
  private readonly datasetService = inject(DatasetService);
  private readonly promptService = inject(PromptService);
  private readonly router = inject(Router);

  form: FormGroup;

  constructor(private fb: FormBuilder) {
    this.form = this.fb.group({
      username: ['', Validators.required],
      gender: ['', Validators.required],
      hobbies: [[]],
      description: ['', Validators.required]
    });
  }

  ngOnInit() {
    this.idleTimeoutService.subscribeIdletimeout();
    this.loadDashboardData();
  }

  ngAfterViewInit() {
    // Any initialization that requires view children
  }

  // Load data from APIs
  loadDashboardData() {
    this.isLoading = true;

    // First, get the total count of datasets (without pagination)
    this.datasetService.getDatasets({
      is_public: true
    }).subscribe({
      next: (response) => {
        this.datasetsCount = response.totalCount;

        // Now get just the most recent 5 datasets for display
        this.datasetService.getDatasets({
          page: 1,
          limit: 5,
          is_public: true
        }).subscribe({
          next: (response) => {
            this.datasets = response.datasets;
          },
          error: (error) => {
            console.error('Error loading recent datasets:', error);
            this.showToast('Failed to load recent datasets', 'error');
          }
        });
      },
      error: (error) => {
        console.error('Error loading dataset count:', error);
        this.showToast('Failed to load dataset count', 'error');
      }
    });

    // Get all prompts to determine total count, and also use for display
    this.promptService.getPrompts({
      is_public: true
    }).subscribe({
      next: (allPrompts) => {
        this.promptsCount = allPrompts.length;

        // Just take the 5 most recent prompts for display
        this.recentPrompts = allPrompts.slice(0, 5);
      },
      error: (error) => {
        console.error('Error loading prompts:', error);
        this.showToast('Failed to load prompts', 'error');
      },
      complete: () => {
        this.isLoading = false;
      }
    });

    // Mock data for evaluations since we don't have that service yet
    this.recentEvaluations = [
      {
        id: 'eval1',
        name: 'Customer Support RAG Evaluation',
        status: 'completed',
        datasetName: 'Customer Support QA',
        promptName: 'Support Query Template',
        modelCount: 3,
        avgRelevance: 0.87,
        avgLatency: 345,
        createdAt: new Date().toISOString()
      },
      {
        id: 'eval2',
        name: 'Medical Context Retrieval Test',
        status: 'running',
        datasetName: 'Medical Context',
        promptName: 'Medical Query Template',
        modelCount: 2,
        createdAt: new Date().toISOString()
      }
    ];
    this.evaluationsCount = 2;
    this.reportsCount = 5;
  }

  // Tab navigation
  setActiveTab(tab: string) {
    this.activeTab = tab;
  }

  // Dataset actions
  createDataset() {
    this.router.navigate(['app/datasets/datasets/upload']);
  }

  viewDataset(dataset: Dataset) {
    this.router.navigate(['app/datasets/datasets', dataset.id]);
  }

  viewAllDatasets() {
    this.router.navigate(['app/datasets']);
  }

  // Prompt actions
  viewPrompt(prompt: PromptResponse) {
    this.router.navigate(['app/prompts', prompt.id]);
  }

  viewAllPrompts() {
    this.router.navigate(['app/prompts']);
  }

  // Evaluation actions
  startNewEvaluation() {
    this.router.navigate(['app/evaluations/create']);
  }

  viewEvaluation(evaluation: EvaluationSummary) {
    this.router.navigate(['app/evaluations', evaluation.id]);
  }

  viewAllEvaluations() {
    this.router.navigate(['app/evaluations']);
  }

  // Report actions
  viewAllReports() {
    this.router.navigate(['app/reports']);
  }

  // Helper functions
  formatFileSize(bytes?: number): string {
    if (!bytes) return '0 B';

    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }

  // Toast notification
  showToast(message: string, type: 'success' | 'error' | 'warning' | 'info' = 'success', title = '') {
    let toastTitle = title;
    if (!title) {
      switch (type) {
        case 'success': toastTitle = 'Success!'; break;
        case 'error': toastTitle = 'Error!'; break;
        case 'warning': toastTitle = 'Warning!'; break;
        case 'info': toastTitle = 'Information'; break;
      }
    }

    if (this.qrscsystoast && this.qrscsystoast.nativeElement) {
      this.qrscsystoast.nativeElement.presentToast(
        message,
        type,
        toastTitle,
        5000
      );
    } else {
      console.log(`${toastTitle}: ${message}`);
    }
  }
}