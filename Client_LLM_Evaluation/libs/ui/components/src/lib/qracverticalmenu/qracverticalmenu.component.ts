import { CommonModule } from '@angular/common';
import { Component, Input, OnInit, AfterViewInit, ElementRef, Renderer2, OnChanges, SimpleChanges, HostListener } from '@angular/core';

interface SubMenu {
  id: string;
  name: string;
  dataurl: string
}
interface SubMenuHeader {
  header: string;
  items: SubMenu[];
}
interface Menu {
  id: string;
  class: string;
  name: string;
  dataurl:string;
  selected: boolean;
  subMenus: SubMenuHeader[];
}
@Component({
 selector: 'qrac-verticalmenu',
 templateUrl: './qracverticalmenu.component.html',
 styleUrls: ['./qracverticalmenu.component.scss'],
 imports: [CommonModule]
})
export class QracVerticalmenuComponent implements OnInit, AfterViewInit {
  @Input() dynamicMenus: Menu[] = [];
  currentMenu: Menu | null = null;
  Issearchmenupnl = false;
  Ishelpmenupnl = false;
  Issettingmenupnl = false;
  menuExpanded = false;
  constructor(private eRef: ElementRef, private renderer: Renderer2) {}
  ngOnInit(): void {
    // No initial selection
  }
  ngAfterViewInit(): void {
    this.initializeTooltips();
  }
  ngOnChanges(changes: SimpleChanges): void {
    if (changes['dynamicMenus'] && !changes['dynamicMenus'].isFirstChange()) {
      this.initializeTooltips();
    }
  }
  initializeTooltips(): void {
    const tooltips = this.eRef.nativeElement.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach((tooltip: HTMLElement) => {
      new (window as any).bootstrap.Tooltip(tooltip);
    });
  }
  hideTooltips(): void {
    const tooltips = this.eRef.nativeElement.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach((tooltip: HTMLElement) => {
      const tooltipInstance = (window as any).bootstrap.Tooltip.getInstance(tooltip);
      if (tooltipInstance) {
        tooltipInstance.hide();
      }
    });
  }
  selectMenu(menu: Menu): void {
    this.hideTooltips();
    this.dynamicMenus.forEach(m => m.selected = false);
    menu.selected = true;
    this.Issearchmenupnl = false;
    this.currentMenu = menu;

    this.menuExpanded = true;
    this.initializeTooltips(); // Reinitialize tooltips after selecting a menu
  }
  selectSubMenu(submenu: SubMenu): void {
    console.log(submenu.id);
    this.hideTooltips();
    this.menuExpanded = false;
  }
  loadTemplate(template: string): void {
    alert();
  }
  resetPanels(): void {
    this.Issearchmenupnl = false;
    this.Ishelpmenupnl = false;
    this.Issettingmenupnl = false;
  }

  TemplateLoader(template: string): void {
    this.hideTooltips();
    this.menuExpanded = true;
    this.currentMenu = null;
    this.resetPanels();
      switch(template) {
        case 'search':
          this.Issearchmenupnl = true;
          break;
        case 'settings':
          this.Ishelpmenupnl = true;
          break;
        case 'help':
          this.Issettingmenupnl = true;
          break;
        default:
           break;
      }
  }
  closeMenu(): void {
    this.menuExpanded = false;
  }
  @HostListener('document:click', ['$event'])
  onClick(event: MouseEvent): void {
    if (this.menuExpanded && !this.eRef.nativeElement.contains(event.target as Node)) {
      this.closeMenu();
    }
  }
  @HostListener('document:keydown.escape', ['$event'])
  onKeydownHandler(event: KeyboardEvent): void {
    if (this.menuExpanded) {
      this.resetMenu();
    }
  }
  resetMenu(): void {
    this.hideTooltips();
    this.dynamicMenus.forEach(m => m.selected = false);
    const selectedMenu = this.dynamicMenus.find(m => m.selected);
    if (selectedMenu) {
      this.selectMenu(selectedMenu);
    } else {
      this.currentMenu = null;
      this.menuExpanded = false;
      document.getElementById('dynamic-template')!.innerHTML = '';
    }
  }
 }
