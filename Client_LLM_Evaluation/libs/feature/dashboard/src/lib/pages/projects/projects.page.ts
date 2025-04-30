import { CommonModule } from '@angular/common';
import { Component, CUSTOM_ELEMENTS_SCHEMA, ElementRef, ViewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { QracCheckboxComponent, QracRadioComponent, QracSelectComponent, QracTagButtonComponent, QracTextBoxComponent } from '@ngtx-apps/ui/components';
import { EditService, FilterService, GridModule, PageService, SelectionService, SortService, ToolbarService, VirtualScrollService } from '@syncfusion/ej2-angular-grids';
import { data } from './data';

import { CategoryService, ChartModule, ColumnSeriesService, DateTimeService, LineSeriesService, MultiColoredLineSeriesService, ParetoSeriesService, SplineAreaSeriesService, SplineSeriesService, StackingLineSeriesService, StepLineSeriesService } from '@syncfusion/ej2-angular-charts';
import { IPointRenderEventArgs } from '@syncfusion/ej2-charts';

@Component({
  selector: 'app-projects',
  templateUrl: './projects.page.html',
  styleUrls: ['./projects.page.scss'],
  providers: [PageService, SortService, SelectionService, FilterService, ToolbarService, EditService, CategoryService, LineSeriesService, StepLineSeriesService, SplineSeriesService, StackingLineSeriesService, DateTimeService,
    SplineAreaSeriesService, MultiColoredLineSeriesService, ParetoSeriesService, ColumnSeriesService, VirtualScrollService],
  imports: [CommonModule, FormsModule, ChartModule, QracTagButtonComponent,
    GridModule],
  schemas:[CUSTOM_ELEMENTS_SCHEMA],
})
export class ProjectsPage {
  public data: any[] = [];
    public pageSettings!: object;
    public selectOptions!: object;
    public filterSettings!: object;
    public toolbar!: string[];
    public editSettings!: object;
    public orderidrules!: object;
    public customeridrules!: object;
    public freightrules!: object;
    public primaryXAxis?: object;
    public chartData?: object[];
    public title?: string;
    public primaryYAxis?: object;
    public marker?: object;
    splineData: object[] = [
      { x: 'D-360', y: 80 },
      { x: 'D-270', y: 60 },
      { x: 'D-150', y: 100 },
      { x: 'D-90', y: 30 },
      { x: 'D-30', y: 20 },
      { x: 'D-15', y: 10 },

  ];

    @ViewChild('qrscsystoast', { read: ElementRef, static: false }) qrscsystoast!: ElementRef;
    @ViewChild('SideModalComponent', { read: ElementRef, static: false }) SideModalComponent!: ElementRef;

    ngOnInit(): void {
        this.data = data;
        console.log(this.data, 'data');
        this.selectOptions = { persistSelection: true };
       // this.pageSettings = { pageSizes: ['All', 10, 20, 50, 100], pageSize: 10 };
        this.filterSettings = { type: 'Excel' };
        this.toolbar = [];
        this.editSettings = { allowEditing: true, allowAdding: true, allowDeleting: true };
        this.orderidrules = { required: true, number: true };
        this.customeridrules = { required: true, minLength: 5 };
        this.freightrules = { required: true, min: 0 };
        this.marker = { visible: true };
        this.chartData = this.splineData;
        this.primaryXAxis = {
           title: 'Checkpoints',
           valueType: 'Category'
        };
        this.title = 'New Year Peak 2025';
    }

    public pointRender(args: IPointRenderEventArgs)  {
      args.fill = '#ff6347';
  }

    onButtonClick(event: Event) {
      console.log('Button clicked:', event);
     }

     opentoast1() {
      this.qrscsystoast.nativeElement.presentToast(
        "Your changes have been saved!",
        "success",
        "Success!",
        5000
    )
    }

    openModal1() {
      this.SideModalComponent.nativeElement.openModal();
    }
    closeModal1() {
      this.SideModalComponent.nativeElement.closeModal();
    }
}
